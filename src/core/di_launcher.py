# src/core/di_launcher.py

"""
Dependency injection-aware launcher for the Ares trading system.

This module provides a launcher that uses proper dependency injection
patterns for creating and managing trading system components.
"""

from src.core.dependency_injection import DependencyContainer
from src.core.enhanced_factories import TradingSystemFactory
from src.core.service_registry import ServiceRegistry
from src.utils.logger import system_logger
from typing import Any
from src.config import CONFIG


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
        self.container = DependencyContainer(self.config)
        self.registry = ServiceRegistry(self.container)

        # Create factory
        self.factory = TradingSystemFactory(self.container)

        # System components
        self.system_components: dict[str, Any] = {}
        self.is_running = False

    def _create_paper_trading_config(self, symbol: str, exchange: str) -> dict[str, Any]:
        """Create configuration for paper trading mode."""
        return {
            "mode": "paper_trading",
            "symbol": symbol,
            "exchange": {
                "name": exchange,
                "testnet": True,
                "paper_trading": True,
            },
            "state": {
                "persistence": "memory",
                "backup_enabled": False,
            },
            "performance": {
                "tracking_enabled": True,
                "reporting_interval": 60,
            },
            "use_modular_components": True,
        }

    def _create_live_trading_config(self, symbol: str, exchange: str) -> dict[str, Any]:
        """Create configuration for live trading mode."""
        return {
            "mode": "live_trading",
            "symbol": symbol,
            "exchange": {
                "name": exchange,
                "testnet": False,
                "paper_trading": False,
            },
            "state": {
                "persistence": "database",
                "backup_enabled": True,
            },
            "performance": {
                "tracking_enabled": True,
                "reporting_interval": 30,
            },
            "use_modular_components": True,
        }

    async def _start_all_components(self) -> None:
        """Start all trading system components."""
        try:
            for name, component in self.system_components.items():
                if hasattr(component, "start"):
                    await component.start()
                    self.logger.info(f"Started component: {name}")

        except Exception as e:
            self.logger.exception(f"Failed to start components: {e}")
            raise
