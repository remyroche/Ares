from datetime import datetime, timedelta
from typing import Any

import numpy as np

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    initialization_error,
    invalid,
    missing,
)


class ModularTactician:
    """
    Enhanced modular tactician with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize modular tactician with enhanced type safety.

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

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid modular tactician configuration"),
            AttributeError: (False, "Missing required tactician parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="modular tactician initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize modular tactician with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        self.logger.info("Initializing Modular Tactician...")

        # Load tactician configuration
        await self._load_tactician_configuration()

        # Validate configuration
        if not self._validate_configuration():
            self.print(invalid("Invalid configuration for modular tactician"))
            return False

        # Initialize tactician modules
        await self._initialize_tactician_modules()

        self.logger.info(
            "✅ Modular Tactician initialization completed successfully",
        )
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician configuration loading",
    )
    async def _load_tactician_configuration(self) -> None:
        """Load tactician configuration."""
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
        self.enable_entry_monitoring = self.tactician_config["enable_entry_monitoring"]
        self.enable_exit_monitoring = self.tactician_config["enable_exit_monitoring"]

        self.logger.info("Tactician configuration loaded successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate tactician configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        # Validate tactician interval
        if self.tactician_interval <= 0:
            self.print(invalid("Invalid tactician interval"))
            return False

        # Validate max tactician history
        if self.max_tactician_history <= 0:
            self.print(invalid("Invalid max tactician history"))
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
            self.print(error("At least one tactician type must be enabled"))
            return False

        self.logger.info("Configuration validation successful")
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician modules initialization",
    )
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

        except Exception:
            self.print(
                initialization_error("Error initializing tactician modules: {e}"),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="entry monitoring initialization",
    )
    async def _initialize_entry_monitoring(self) -> None:
        """Initialize entry monitoring module."""
        try:
            # Initialize entry strategies
            self.entry_strategies = {
                "breakout": True,
                "pullback": True,
                "momentum": True,
                "mean_reversion": True,
            }

            self.logger.info("Entry monitoring module initialized")

        except Exception:
            self.print(initialization_error("Error initializing entry monitoring: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="exit monitoring initialization",
    )
    async def _initialize_exit_monitoring(self) -> None:
        """Initialize exit monitoring module."""
        try:
            # Initialize exit strategies
            self.exit_strategies = {
                "stop_loss": True,
                "take_profit": True,
                "trailing_stop": True,
                "time_based": True,
            }

            self.logger.info("Exit monitoring module initialized")

        except Exception:
            self.print(initialization_error("Error initializing exit monitoring: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position monitoring initialization",
    )
    async def _initialize_position_monitoring(self) -> None:
        """Initialize position monitoring module."""
        try:
            # Initialize position strategies
            self.position_strategies = {
                "scaling": True,
                "averaging": True,
                "hedging": True,
                "rebalancing": True,
            }

            self.logger.info("Position monitoring module initialized")

        except Exception:
            self.print(
                initialization_error("Error initializing position monitoring: {e}"),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk monitoring initialization",
    )
    async def _initialize_risk_monitoring(self) -> None:
        """Initialize risk monitoring module."""
        try:
            # Initialize risk strategies
            self.risk_strategies = {
                "position_sizing": True,
                "leverage_control": True,
                "correlation_monitoring": True,
                "volatility_adjustment": True,
            }

            self.logger.info("Risk monitoring module initialized")

        except Exception:
            self.print(initialization_error("Error initializing risk monitoring: {e}"))

    @handle_specific_errors(
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
        """
        Execute tactician monitoring.

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
            self.logger.info("🔄 Starting tactician monitoring...")

            # Perform entry monitoring
            if self.enable_entry_monitoring:
                entry_results = await self._perform_entry_monitoring(
                    market_data,
                    strategy_data,
                )
                self.tactician_results["entry"] = entry_results

            # Perform exit monitoring
            if self.enable_exit_monitoring:
                exit_results = await self._perform_exit_monitoring(
                    market_data,
                    strategy_data,
                )
                self.tactician_results["exit"] = exit_results

            # Perform position monitoring
            if self.tactician_config.get("enable_position_monitoring", False):
                position_results = await self._perform_position_monitoring(
                    market_data,
                    strategy_data,
                )
                self.tactician_results["position"] = position_results

            # Perform risk monitoring
            if self.tactician_config.get("enable_risk_monitoring", True):
                risk_results = await self._perform_risk_monitoring(
                    market_data,
                    strategy_data,
                )
                self.tactician_results["risk"] = risk_results

            # Store tactician results
            await self._store_tactician_results()

            self.is_tactician_active = False
            self.logger.info("✅ Tactician monitoring completed successfully")
            return True

        except Exception:
            self.print(error("Error executing tactician: {e}"))
            self.is_tactician_active = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="tactician inputs validation",
    )
    def _validate_tactician_inputs(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """
        Validate tactician inputs.

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
                    self.print(missing("Missing required market data field: {field}"))
                    return False

            # Check required strategy data fields
            required_strategy_fields = ["signal", "position_size", "timestamp"]
            for field in required_strategy_fields:
                if field not in strategy_data:
                    self.print(missing("Missing required strategy data field: {field}"))
                    return False

            # Validate data types
            if not isinstance(market_data["price"], int | float):
                self.print(invalid("Invalid price data type"))
                return False

            if not isinstance(strategy_data["position_size"], int | float):
                self.print(invalid("Invalid position size data type"))
                return False

            return True

        except Exception:
            self.print(error("Error validating tactician inputs: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="entry monitoring",
    )
    async def _perform_entry_monitoring(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform entry monitoring.

        Args:
            market_data: Market data dictionary
            strategy_data: Strategy data dictionary

        Returns:
            Dict[str, Any]: Entry monitoring results
        """
        try:
            results = {}

            # Check breakout entry
            if self.entry_strategies.get("breakout", False):
                results["breakout"] = self._check_breakout_entry(
                    market_data,
                    strategy_data,
                )

            # Check pullback entry
            if self.entry_strategies.get("pullback", False):
                results["pullback"] = self._check_pullback_entry(
                    market_data,
                    strategy_data,
                )

            # Check momentum entry
            if self.entry_strategies.get("momentum", False):
                results["momentum"] = self._check_momentum_entry(
                    market_data,
                    strategy_data,
                )

            # Check mean reversion entry
            if self.entry_strategies.get("mean_reversion", False):
                results["mean_reversion"] = self._check_mean_reversion_entry(
                    market_data,
                    strategy_data,
                )

            self.logger.info("Entry monitoring completed")
            return results

        except Exception:
            self.print(error("Error performing entry monitoring: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="exit monitoring",
    )
    async def _perform_exit_monitoring(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform exit monitoring.

        Args:
            market_data: Market data dictionary
            strategy_data: Strategy data dictionary

        Returns:
            Dict[str, Any]: Exit monitoring results
        """
        try:
            results = {}

            # Check stop loss exit
            if self.exit_strategies.get("stop_loss", False):
                results["stop_loss"] = self._check_stop_loss_exit(
                    market_data,
                    strategy_data,
                )

            # Check take profit exit
            if self.exit_strategies.get("take_profit", False):
                results["take_profit"] = self._check_take_profit_exit(
                    market_data,
                    strategy_data,
                )

            # Check trailing stop exit
            if self.exit_strategies.get("trailing_stop", False):
                results["trailing_stop"] = self._check_trailing_stop_exit(
                    market_data,
                    strategy_data,
                )

            # Check time based exit
            if self.exit_strategies.get("time_based", False):
                results["time_based"] = self._check_time_based_exit(
                    market_data,
                    strategy_data,
                )

            self.logger.info("Exit monitoring completed")
            return results

        except Exception:
            self.print(error("Error performing exit monitoring: {e}"))
            return {}

    # Entry monitoring calculation methods
    def _check_breakout_entry(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check for breakout entry opportunity."""
        try:
            # Simulate breakout entry check
            current_price = market_data.get("price", 0)
            resistance_level = current_price * 1.02  # 2% above current price

            return current_price > resistance_level
        except Exception:
            self.print(error("Error checking breakout entry: {e}"))
            return False

    def _check_pullback_entry(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check for pullback entry opportunity."""
        try:
            # Simulate pullback entry check
            current_price = market_data.get("price", 0)
            support_level = current_price * 0.98  # 2% below current price

            return current_price < support_level
        except Exception:
            self.print(error("Error checking pullback entry: {e}"))
            return False

    def _check_momentum_entry(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check for momentum entry opportunity."""
        try:
            # Simulate momentum entry check
            signal = strategy_data.get("signal", "HOLD")

            return signal in ["BUY", "SELL"]
        except Exception:
            self.print(error("Error checking momentum entry: {e}"))
            return False

    def _check_mean_reversion_entry(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check for mean reversion entry opportunity."""
        try:
            # Simulate mean reversion entry check
            current_price = market_data.get("price", 0)
            avg_price = current_price * 1.01  # Simulated average price

            deviation = abs(current_price - avg_price) / avg_price

            return deviation > 0.05  # 5% deviation threshold
        except Exception:
            self.print(error("Error checking mean reversion entry: {e}"))
            return False

    # Exit monitoring calculation methods
    def _check_stop_loss_exit(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check for stop loss exit."""
        try:
            # Simulate stop loss exit check
            current_price = market_data.get("price", 0)
            entry_price = current_price * 1.01  # Simulated entry price
            stop_loss_pct = 0.02  # 2% stop loss

            loss_pct = (current_price - entry_price) / entry_price

            return loss_pct < -stop_loss_pct
        except Exception:
            self.print(error("Error checking stop loss exit: {e}"))
            return False

    def _check_take_profit_exit(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check for take profit exit."""
        try:
            # Simulate take profit exit check
            current_price = market_data.get("price", 0)
            entry_price = current_price * 0.99  # Simulated entry price
            take_profit_pct = 0.04  # 4% take profit

            profit_pct = (current_price - entry_price) / entry_price

            return profit_pct > take_profit_pct
        except Exception:
            self.print(error("Error checking take profit exit: {e}"))
            return False

    def _check_trailing_stop_exit(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check for trailing stop exit."""
        try:
            # Simulate trailing stop exit check
            current_price = market_data.get("price", 0)
            highest_price = current_price * 1.03  # Simulated highest price
            trailing_pct = 0.015  # 1.5% trailing stop

            drawdown = (highest_price - current_price) / highest_price

            return drawdown > trailing_pct
        except Exception:
            self.print(error("Error checking trailing stop exit: {e}"))
            return False

    def _check_time_based_exit(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check for time based exit."""
        try:
            # Simulate time based exit check
            entry_time = datetime.now() - timedelta(hours=2)  # Simulated entry time
            max_hold_time = timedelta(hours=4)  # 4 hour max hold time

            current_time = datetime.now()
            hold_time = current_time - entry_time

            return hold_time > max_hold_time
        except Exception:
            self.print(error("Error checking time based exit: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position monitoring",
    )
    async def _perform_position_monitoring(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform position monitoring.

        Args:
            market_data: Market data dictionary
            strategy_data: Strategy data dictionary

        Returns:
            Dict[str, Any]: Position monitoring results
        """
        try:
            results = {}

            # Check scaling opportunities
            if self.position_strategies.get("scaling", False):
                results["scaling"] = self._check_scaling_opportunity(
                    market_data,
                    strategy_data,
                )

            # Check averaging opportunities
            if self.position_strategies.get("averaging", False):
                results["averaging"] = self._check_averaging_opportunity(
                    market_data,
                    strategy_data,
                )

            # Check hedging opportunities
            if self.position_strategies.get("hedging", False):
                results["hedging"] = self._check_hedging_opportunity(
                    market_data,
                    strategy_data,
                )

            # Check rebalancing opportunities
            if self.position_strategies.get("rebalancing", False):
                results["rebalancing"] = self._check_rebalancing_opportunity(
                    market_data,
                    strategy_data,
                )

            self.logger.info("Position monitoring completed")
            return results

        except Exception:
            self.print(error("Error performing position monitoring: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="risk monitoring",
    )
    async def _perform_risk_monitoring(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform risk monitoring.

        Args:
            market_data: Market data dictionary
            strategy_data: Strategy data dictionary

        Returns:
            Dict[str, Any]: Risk monitoring results
        """
        try:
            results = {}

            # Check position sizing
            if self.risk_strategies.get("position_sizing", False):
                results["position_sizing"] = self._check_position_sizing(
                    market_data,
                    strategy_data,
                )

            # Check leverage control
            if self.risk_strategies.get("leverage_control", False):
                results["leverage_control"] = self._check_leverage_control(
                    market_data,
                    strategy_data,
                )

            # Check correlation monitoring
            if self.risk_strategies.get("correlation_monitoring", False):
                results["correlation_monitoring"] = self._check_correlation_monitoring(
                    market_data,
                    strategy_data,
                )

            # Check volatility adjustment
            if self.risk_strategies.get("volatility_adjustment", False):
                results["volatility_adjustment"] = self._check_volatility_adjustment(
                    market_data,
                    strategy_data,
                )

            self.logger.info("Risk monitoring completed")
            return results

        except Exception:
            self.print(error("Error performing risk monitoring: {e}"))
            return {}

    # Position monitoring calculation methods
    def _check_scaling_opportunity(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check for scaling opportunity."""
        try:
            # Simulate scaling opportunity check
            current_price = market_data.get("price", 0)
            entry_price = current_price * 0.98  # Simulated entry price

            profit_pct = (current_price - entry_price) / entry_price

            return profit_pct > 0.03  # 3% profit threshold for scaling
        except Exception:
            self.print(error("Error checking scaling opportunity: {e}"))
            return False

    def _check_averaging_opportunity(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check for averaging opportunity."""
        try:
            # Simulate averaging opportunity check
            current_price = market_data.get("price", 0)
            entry_price = current_price * 1.02  # Simulated entry price

            loss_pct = (current_price - entry_price) / entry_price

            return loss_pct < -0.02  # 2% loss threshold for averaging
        except Exception:
            self.print(error("Error checking averaging opportunity: {e}"))
            return False

    def _check_hedging_opportunity(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check for hedging opportunity."""
        try:
            # Simulate hedging opportunity check
            volatility = np.random.random() * 0.05  # Random volatility
            high_volatility_threshold = 0.03  # 3% volatility threshold

            return volatility > high_volatility_threshold
        except Exception:
            self.print(error("Error checking hedging opportunity: {e}"))
            return False

    def _check_rebalancing_opportunity(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check for rebalancing opportunity."""
        try:
            # Simulate rebalancing opportunity check
            current_allocation = 0.6  # Simulated current allocation
            target_allocation = 0.5  # Simulated target allocation
            rebalance_threshold = 0.1  # 10% threshold

            allocation_deviation = abs(current_allocation - target_allocation)

            return allocation_deviation > rebalance_threshold
        except Exception:
            self.print(error("Error checking rebalancing opportunity: {e}"))
            return False

    # Risk monitoring calculation methods
    def _check_position_sizing(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check position sizing."""
        try:
            # Simulate position sizing check
            position_size = strategy_data.get("position_size", 0)
            max_position_size = 0.25  # 25% max position size

            return position_size > max_position_size
        except Exception:
            self.print(error("Error checking position sizing: {e}"))
            return False

    def _check_leverage_control(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check leverage control."""
        try:
            # Simulate leverage control check
            current_leverage = 2.5  # Simulated current leverage
            max_leverage = 3.0  # 3x max leverage

            return current_leverage > max_leverage
        except Exception:
            self.print(error("Error checking leverage control: {e}"))
            return False

    def _check_correlation_monitoring(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check correlation monitoring."""
        try:
            # Simulate correlation monitoring check
            correlation = (
                np.random.random() * 2 - 1
            )  # Random correlation between -1 and 1
            high_correlation_threshold = 0.8  # 80% correlation threshold

            return abs(correlation) > high_correlation_threshold
        except Exception:
            self.print(error("Error checking correlation monitoring: {e}"))
            return False

    def _check_volatility_adjustment(
        self,
        market_data: dict[str, Any],
        strategy_data: dict[str, Any],
    ) -> bool:
        """Check volatility adjustment."""
        try:
            # Simulate volatility adjustment check
            current_volatility = np.random.random() * 0.1  # Random volatility
            volatility_threshold = 0.05  # 5% volatility threshold

            return current_volatility > volatility_threshold
        except Exception:
            self.print(error("Error checking volatility adjustment: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician results storage",
    )
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

        except Exception:
            self.print(error("Error storing tactician results: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician results getting",
    )
    def get_tactician_results(
        self,
        tactician_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get tactician results.

        Args:
            tactician_type: Optional tactician type filter

        Returns:
            Dict[str, Any]: Tactician results
        """
        try:
            if tactician_type:
                return self.tactician_results.get(tactician_type, {})
            return self.tactician_results.copy()

        except Exception:
            self.print(error("Error getting tactician results: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="tactician history getting",
    )
    def get_tactician_history(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get tactician history.

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

        except Exception:
            self.print(error("Error getting tactician history: {e}"))
            return []

    def get_tactician_status(self) -> dict[str, Any]:
        """
        Get tactician status information.

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

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="modular tactician cleanup",
    )
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

        except Exception:
            self.print(error("Error stopping modular tactician: {e}"))


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
    """
    Setup global modular tactician.

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
