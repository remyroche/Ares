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
from typing import Any, Optional
from src.config import CONFIG


class DILauncher:
    """
    Dependency injection-aware launcher for the Ares trading system.
    
    This launcher creates and manages trading system components using
    proper dependency injection patterns.
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize the DI Launcher with configuration."""
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
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize DILauncher."""
        try:
            self.logger.info("🚀 Initializing DILauncher...")
            self.is_initialized = True
            self.logger.info("✅ DILauncher initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing DILauncher: {e}")
            return False
    
    async def launch_paper_trading(self, symbol: str, exchange: str) -> dict[str, Any]:
        """Launch paper trading system for the specified symbol and exchange."""
        try:
            self.logger.info(f"Launching paper trading for {symbol} on {exchange}")
            
            # Configure for paper trading
            trading_config = self._create_paper_trading_config(symbol, exchange)
            
            # Register services
            self.registry.register_all_services(trading_config)
            
            # Create exchange client
            from src.exchange.binance import BinanceClient
            exchange_client = BinanceClient(trading_config.get("exchange", {}))
            
            # Create state manager
            from src.utils.state_manager import StateManager
            state_manager = StateManager(trading_config.get("state", {}))
            
            # Create performance reporter
            from src.supervisor.performance_reporter import PerformanceReporter
            performance_reporter = PerformanceReporter(trading_config.get("performance", {}))
            
            # Create trading components
            self.system_components = await self.factory.create_complete_trading_system(
                exchange_client,
                state_manager,
                performance_reporter,
            )
            
            # Start all components
            await self._start_all_components()
            
            self.is_running = True
            self.logger.info("Paper trading system launched successfully")
            
            return self.system_components
            
        except Exception as e:
            self.logger.exception(f"Failed to launch paper trading: {e}")
            raise
    
    async def launch_live_trading(self, symbol: str, exchange: str) -> dict[str, Any]:
        """Launch live trading system for the specified symbol and exchange."""
        try:
            self.logger.info(f"Launching live trading for {symbol} on {exchange}")
            
            # Configure for live trading
            trading_config = self._create_live_trading_config(symbol, exchange)
            
            # Register services
            self.registry.register_all_services(trading_config)
            
            # Create exchange client
            from src.exchange.binance import BinanceClient
            exchange_client = BinanceClient(trading_config.get("exchange", {}))
            
            # Create state manager
            from src.utils.state_manager import StateManager
            state_manager = StateManager(trading_config.get("state", {}))
            
            # Create performance reporter
            from src.supervisor.performance_reporter import PerformanceReporter
            performance_reporter = PerformanceReporter(trading_config.get("performance", {}))
            
            # Create trading components
            self.system_components = await self.factory.create_complete_trading_system(
                exchange_client,
                state_manager,
                performance_reporter,
            )
            
            # Start all components
            await self._start_all_components()
            
            self.is_running = True
            self.logger.info("Live trading system launched successfully")
            
            return self.system_components
            
        except Exception as e:
            self.logger.exception(f"Failed to launch live trading: {e}")
            raise
    
    def _create_paper_trading_config(self, symbol: str, exchange: str) -> dict[str, Any]:
        """Create configuration for paper trading."""
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
        """Create configuration for live trading."""
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
        """Start all system components."""
        try:
            for name, component in self.system_components.items():
                if hasattr(component, "start"):
                    await component.start()
                    self.logger.info(f"Started component: {name}")
        except Exception as e:
            self.logger.exception(f"Failed to start components: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the trading system and all components."""
        try:
            for name, component in self.system_components.items():
                if hasattr(component, "stop"):
                    await component.stop()
                    self.logger.info(f"Stopped component: {name}")
            
            self.is_running = False
            self.logger.info("Trading system stopped")
            
        except Exception as e:
            self.logger.exception(f"Failed to stop components: {e}")
            raise
    
    def get_status(self) -> dict[str, Any]:
        """Get the current status of the trading system."""
        return {
            "is_running": self.is_running,
            "is_initialized": self.is_initialized,
            "components": list(self.system_components.keys()),
            "config": self.config,
        }
