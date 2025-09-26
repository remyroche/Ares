"""Dependency injection aware launcher for the trading system."""

from __future__ import annotations

from typing import Any

from exchange.binance import BinanceClient
from src.config import CONFIG
from src.trading.reporting.performance_reporter import PerformanceReporter
from src.utils.logger import system_logger
from src.utils.state_manager import StateManager

from .dependency_injection import DependencyContainer
from .enhanced_factories import TradingSystemFactory
from .service_registry import ServiceRegistry


class DILauncher:
    """Create and manage trading system components using dependency injection."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or CONFIG
        self.logger = system_logger.getChild("DILauncher")
        self.container = DependencyContainer(self.config)
        self.registry = ServiceRegistry(self.container)
        self.factory = TradingSystemFactory(self.container)
        self.system_components: dict[str, Any] = {}
        self.is_running = False

    async def launch_paper_trading(self, symbol: str, exchange: str) -> dict[str, Any]:
        """Launch the paper-trading configuration."""

        return await self._launch(symbol, exchange, paper_trading=True)

    async def launch_live_trading(self, symbol: str, exchange: str) -> dict[str, Any]:
        """Launch the live-trading configuration."""

        return await self._launch(symbol, exchange, paper_trading=False)

    async def _launch(self, symbol: str, exchange: str, *, paper_trading: bool) -> dict[str, Any]:
        mode = "paper_trading" if paper_trading else "live_trading"

        try:
            self.logger.info("Launching %s for %s on %s", mode, symbol, exchange)
            trading_config = self._build_trading_config(symbol, exchange, paper_trading)
            self.registry.register_all_services(trading_config)

            exchange_client = BinanceClient(trading_config.get("exchange", {}))
            state_manager = StateManager(trading_config.get("state", {}))
            performance_reporter = PerformanceReporter(trading_config.get("performance", {}))

            self.system_components = await self.factory.create_complete_trading_system(
                exchange_client,
                state_manager,
                performance_reporter,
            )
            await self._start_all_components()

            self.is_running = True
            self.logger.info("%s system launched successfully", mode.replace("_", " "))
            return self.system_components
        except Exception as exc:  # pragma: no cover - orchestration guard
            self.logger.exception("Failed to launch %s: %s", mode, exc)
            raise

    def _build_trading_config(self, symbol: str, exchange: str, paper_trading: bool) -> dict[str, Any]:
        return {
            "mode": "paper_trading" if paper_trading else "live_trading",
            "symbol": symbol,
            "exchange": {
                "name": exchange,
                "testnet": paper_trading,
                "paper_trading": paper_trading,
            },
            "state": {
                "persistence": "memory" if paper_trading else "database",
                "backup_enabled": not paper_trading,
            },
            "performance": {
                "tracking_enabled": True,
                "reporting_interval": 60 if paper_trading else 30,
            },
            "use_modular_components": True,
        }

    async def _start_all_components(self) -> None:
        for name, component in self.system_components.items():
            if hasattr(component, "start"):
                await component.start()
                self.logger.info("Started component: %s", name)

    async def stop(self) -> None:
        for name, component in self.system_components.items():
            if hasattr(component, "stop"):
                await component.stop()
                self.logger.info("Stopped component: %s", name)

        self.is_running = False
        self.logger.info("Trading system stopped")

    def get_status(self) -> dict[str, Any]:
        """Return the current launcher status."""

        return {
            "is_running": self.is_running,
            "components": list(self.system_components.keys()),
            "config": self.config,
        }

