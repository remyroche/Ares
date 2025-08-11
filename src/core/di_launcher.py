# src/core/di_launcher.py

"""
Dependency injection-aware launcher for the Ares trading system.

This module provides a launcher that uses proper dependency injection
patterns for creating and managing trading system components.
"""

from typing import Any

from src.config import CONFIG
from src.core.dependency_injection import AsyncServiceContainer
from src.core.enhanced_factories import ComprehensiveFactory
from src.core.service_registry import create_configured_container
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
    warning,
)


class DILauncher:
    """
    Dependency injection-aware launcher for the Ares trading system.

    This launcher creates and manages trading system components using
    proper dependency injection patterns.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or CONFIG
        self.logger = system_logger.getChild("DILauncher")

        # Create DI container with configuration
        self.container = create_configured_container(self.config)

        # Create comprehensive factory
        self.factory = ComprehensiveFactory(self.container)

        # System components
        self.system_components: dict[str, Any] = {}
        self.is_running = False

    async def launch_paper_trading(self, symbol: str, exchange: str) -> dict[str, Any]:
        """
        Launch paper trading mode with dependency injection.

        Args:
            symbol: Trading symbol (e.g., ETHUSDT)
            exchange: Exchange name (e.g., BINANCE)

        Returns:
            Dictionary containing system components
        """
        try:
            self.logger.info(f"Launching paper trading for {symbol} on {exchange}")

            # Configure for paper trading
            trading_config = self._create_paper_trading_config(symbol, exchange)

            # Create full system
            self.system_components = await self.factory.create_full_system(
                trading_config,
            )

            # Start all components
            await self._start_all_components()

            self.is_running = True
            self.logger.info("Paper trading system launched successfully")

            return self.system_components

        except Exception:
            self.print(failed("Failed to launch paper trading: {e}"))
            raise

    async def launch_live_trading(self, symbol: str, exchange: str) -> dict[str, Any]:
        """
        Launch live trading mode with dependency injection.

        Args:
            symbol: Trading symbol (e.g., ETHUSDT)
            exchange: Exchange name (e.g., BINANCE)

        Returns:
            Dictionary containing system components
        """
        try:
            self.logger.info(f"Launching live trading for {symbol} on {exchange}")

            # Configure for live trading
            trading_config = self._create_live_trading_config(symbol, exchange)

            # Create full system
            self.system_components = await self.factory.create_full_system(
                trading_config,
            )

            # Start all components
            await self._start_all_components()

            self.is_running = True
            self.logger.info("Live trading system launched successfully")

            return self.system_components

        except Exception:
            self.print(failed("Failed to launch live trading: {e}"))
            raise

    async def launch_backtesting(self, symbol: str, exchange: str) -> dict[str, Any]:
        """
        Launch backtesting mode with dependency injection.

        Args:
            symbol: Trading symbol (e.g., ETHUSDT)
            exchange: Exchange name (e.g., BINANCE)

        Returns:
            Dictionary containing system components
        """
        try:
            self.logger.info(f"Launching backtesting for {symbol} on {exchange}")

            # Configure for backtesting
            trading_config = self._create_backtesting_config(symbol, exchange)

            # Create full system
            self.system_components = await self.factory.create_full_system(
                trading_config,
            )

            # Initialize components (but don't start them for backtesting)
            await self._initialize_all_components()

            self.logger.info("Backtesting system launched successfully")

            return self.system_components

        except Exception:
            self.print(failed("Failed to launch backtesting: {e}"))
            raise

    async def launch_training(self, symbol: str, exchange: str) -> dict[str, Any]:
        """
        Launch model training mode with dependency injection.

        Args:
            symbol: Trading symbol (e.g., ETHUSDT)
            exchange: Exchange name (e.g., BINANCE)

        Returns:
            Dictionary containing system components
        """
        try:
            self.logger.info(f"Launching model training for {symbol} on {exchange}")

            # Configure for training
            training_config = self._create_training_config(symbol, exchange)

            # Create training-specific system
            self.system_components = await self._create_training_system(training_config)

            self.logger.info("Training system launched successfully")

            return self.system_components

        except Exception:
            self.print(failed("Failed to launch training: {e}"))
            raise

    def _create_paper_trading_config(
        self,
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """Create configuration for paper trading."""
        config = self.config.copy()
        config.update(
            {
                "trading_mode": "paper",
                "symbol": symbol,
                "exchange": {
                    "name": exchange.lower(),
                    "environment": "paper",
                    **config.get("exchange", {}),
                },
                "use_modular_components": True,
            },
        )
        return config

    def _create_live_trading_config(self, symbol: str, exchange: str) -> dict[str, Any]:
        """Create configuration for live trading."""
        config = self.config.copy()
        config.update(
            {
                "trading_mode": "live",
                "symbol": symbol,
                "exchange": {
                    "name": exchange.lower(),
                    "environment": "live",
                    **config.get("exchange", {}),
                },
                "use_modular_components": True,
            },
        )
        return config

    def _create_backtesting_config(self, symbol: str, exchange: str) -> dict[str, Any]:
        """Create configuration for backtesting."""
        config = self.config.copy()
        config.update(
            {
                "trading_mode": "backtest",
                "symbol": symbol,
                "exchange": {
                    "name": exchange.lower(),
                    "environment": "backtest",
                    **config.get("exchange", {}),
                },
                "use_modular_components": True,
            },
        )
        return config

    def _create_training_config(self, symbol: str, exchange: str) -> dict[str, Any]:
        """Create configuration for model training."""
        config = self.config.copy()
        config.update(
            {
                "trading_mode": "training",
                "symbol": symbol,
                "exchange": {
                    "name": exchange.lower(),
                    "environment": "training",
                    **config.get("exchange", {}),
                },
                "use_modular_components": False,  # Use training-specific components
                "training": {
                    "enable_model_training": True,
                    "enable_hyperparameter_optimization": True,
                    **config.get("training", {}),
                },
            },
        )
        return config

    async def _create_training_system(self, config: dict[str, Any]) -> dict[str, Any]:
        """Create training-specific system components."""
        try:
            # Create training manager using DI
            from src.training.training_manager import TrainingManager

            training_manager = self.container.resolve(TrainingManager)

            # Create minimal system for training
            db_manager = self.factory.database_factory.create_database_manager(config)

            return {
                "training_manager": training_manager,
                "db_manager": db_manager,
                "container": self.container,
            }

        except Exception:
            self.print(failed("Failed to create training system: {e}"))
            raise

    async def _start_all_components(self) -> None:
        """Start all system components."""
        try:
            self.logger.info("Starting all system components")

            # Start components in dependency order
            start_order = ["analyst", "strategist", "tactician", "supervisor"]

            for component_name in start_order:
                component = self.system_components.get(component_name)
                if component and hasattr(component, "start"):
                    await component.start()
                    self.logger.debug(f"Started {component_name}")

            self.logger.info("All components started successfully")

        except Exception:
            self.print(failed("Failed to start components: {e}"))
            raise

    async def _initialize_all_components(self) -> None:
        """Initialize all system components without starting them."""
        try:
            self.logger.info("Initializing all system components")

            # Initialize components in dependency order
            init_order = ["analyst", "strategist", "tactician", "supervisor"]

            for component_name in init_order:
                component = self.system_components.get(component_name)
                if component and hasattr(component, "initialize"):
                    success = await component.initialize()
                    if not success:
                        msg = f"Failed to initialize {component_name}"
                        raise RuntimeError(msg)
                    self.logger.debug(f"Initialized {component_name}")

            self.logger.info("All components initialized successfully")

        except Exception:
            self.print(failed("Failed to initialize components: {e}"))
            raise

    async def stop_system(self) -> None:
        """Stop the trading system."""
        try:
            if not self.is_running:
                self.print(warning("System is not running"))
                return

            self.logger.info("Stopping trading system")

            # Stop components in reverse dependency order
            stop_order = ["supervisor", "tactician", "strategist", "analyst"]

            for component_name in stop_order:
                component = self.system_components.get(component_name)
                if component and hasattr(component, "stop"):
                    await component.stop()
                    self.logger.debug(f"Stopped {component_name}")

            self.is_running = False
            self.logger.info("Trading system stopped successfully")

        except Exception:
            self.print(failed("Failed to stop system: {e}"))
            raise

    def get_system_info(self) -> dict[str, Any]:
        """Get information about the current system state."""
        return {
            "is_running": self.is_running,
            "components": list(self.system_components.keys()),
            "container_info": self.factory.get_container_info(),
            "config_summary": {
                "trading_mode": self.config.get("trading_mode"),
                "symbol": self.config.get("symbol"),
                "exchange": self.config.get("exchange", {}).get("name"),
                "use_modular_components": self.config.get("use_modular_components"),
            },
        }

    def get_component(self, component_name: str) -> Any:
        """Get a specific component by name."""
        return self.system_components.get(component_name)

    def get_container(self) -> AsyncServiceContainer:
        """Get the dependency injection container."""
        return self.container


# Convenience functions for quick launching
async def launch_paper_trading(
    symbol: str,
    exchange: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Quick launch function for paper trading."""
    launcher = DILauncher(config)
    return await launcher.launch_paper_trading(symbol, exchange)


async def launch_live_trading(
    symbol: str,
    exchange: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Quick launch function for live trading."""
    launcher = DILauncher(config)
    return await launcher.launch_live_trading(symbol, exchange)


async def launch_backtesting(
    symbol: str,
    exchange: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Quick launch function for backtesting."""
    launcher = DILauncher(config)
    return await launcher.launch_backtesting(symbol, exchange)


async def launch_training(
    symbol: str,
    exchange: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Quick launch function for model training."""
    launcher = DILauncher(config)
    return await launcher.launch_training(symbol, exchange)
