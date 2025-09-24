"""
Unified Trading System

Main system that integrates live trading, exchange connections, and order management.
Provides a complete trading solution with multi-exchange support.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
import logging

from .config import TradingConfig, TradingMode
from .trading_orchestrator import TradingOrchestrator, TradingSignal
from ..exchanges import TradingReceiver
from ..exchange.factory import ExchangeFactory
from ..src.interfaces.base_interfaces import TradeDecision, AnalysisResult, StrategyResult


@dataclass
class SystemConfig:
    """System configuration"""
    trading_config: TradingConfig
    exchanges: List[str] = field(default_factory=lambda: ["binance"])
    enable_websockets: bool = True
    enable_paper_trading: bool = True
    log_level: str = "INFO"
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class UnifiedTradingSystem:
    """Unified system for live trading across multiple exchanges"""

    def __init__(self, system_config: SystemConfig):
        self.system_config = system_config
        self.logger = logging.getLogger(__name__)

        # Core components
        self.trading_receiver: Optional[TradingReceiver] = None
        self.trading_orchestrator: Optional[TradingOrchestrator] = None

        # System state
        self._running = False
        self._initialized = False
        self.start_time: Optional[datetime] = None

        # Event handlers
        self.system_handlers: Dict[str, List[Callable[[Any], Awaitable[None]]]] = {
            "on_system_start": [],
            "on_system_stop": [],
            "on_trading_start": [],
            "on_trading_stop": [],
            "on_error": []
        }

        # Performance tracking
        self.system_stats = {
            "uptime": 0.0,
            "total_signals_processed": 0,
            "total_trades_executed": 0,
            "total_exchanges_connected": 0,
            "errors": 0,
            "warnings": 0
        }

    async def initialize(self) -> None:
        """Initialize the unified trading system"""
        if self._initialized:
            return

        self.logger.info("Initializing unified trading system...")

        try:
            # Configure logging
            logging.basicConfig(
                level=getattr(logging, self.system_config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            # Initialize trading receiver
            await self._initialize_trading_receiver()

            # Initialize trading orchestrator
            await self._initialize_trading_orchestrator()

            # Configure exchanges
            await self._configure_exchanges()

            self._initialized = True
            self.logger.info("Unified trading system initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            await self._notify_handlers("on_error", {"type": "initialization_error", "error": str(e)})
            raise

    async def start(self) -> None:
        """Start the unified trading system"""
        if self._running:
            return

        if not self._initialized:
            await self.initialize()

        self.logger.info("Starting unified trading system...")

        try:
            self.start_time = datetime.now()

            # Start trading orchestrator
            if self.trading_orchestrator:
                await self.trading_orchestrator.start()

            self._running = True

            # Notify handlers
            await self._notify_handlers("on_system_start", {"timestamp": datetime.now()})
            await self._notify_handlers("on_trading_start", {"timestamp": datetime.now()})

            self.logger.info("Unified trading system started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            await self._notify_handlers("on_error", {"type": "start_error", "error": str(e)})
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the unified trading system"""
        if not self._running:
            return

        self.logger.info("Stopping unified trading system...")

        try:
            self._running = False

            # Stop trading orchestrator
            if self.trading_orchestrator:
                await self.trading_orchestrator.stop()

            # Calculate uptime
            if self.start_time:
                self.system_stats["uptime"] = (datetime.now() - self.start_time).total_seconds()

            # Notify handlers
            await self._notify_handlers("on_trading_stop", {"timestamp": datetime.now()})
            await self._notify_handlers("on_system_stop", {"timestamp": datetime.now()})

            self.logger.info("Unified trading system stopped")

        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
            await self._notify_handlers("on_error", {"type": "stop_error", "error": str(e)})

    def register_handler(self, event_type: str, handler: Callable[[Any], Awaitable[None]]) -> None:
        """Register system event handler"""
        if event_type in self.system_handlers:
            self.system_handlers[event_type].append(handler)
            self.logger.info(f"Registered {event_type} handler")

    async def submit_trading_signal(self, signal: TradingSignal) -> bool:
        """Submit a trading signal"""
        if not self._running or not self.trading_orchestrator:
            self.logger.warning("System not running or orchestrator not available")
            return False

        try:
            success = await self.trading_orchestrator.submit_signal(signal)
            if success:
                self.system_stats["total_signals_processed"] += 1
            return success

        except Exception as e:
            self.logger.error(f"Error submitting trading signal: {e}")
            self.system_stats["errors"] += 1
            await self._notify_handlers("on_error", {"type": "signal_error", "error": str(e)})
            return False

    async def execute_trade_decision(self, decision: TradeDecision) -> bool:
        """Execute a trade decision"""
        if not self._running or not self.trading_orchestrator:
            self.logger.warning("System not running or orchestrator not available")
            return False

        try:
            success = await self.trading_orchestrator.execute_trade_decision(decision)
            if success:
                self.system_stats["total_trades_executed"] += 1
            return success

        except Exception as e:
            self.logger.error(f"Error executing trade decision: {e}")
            self.system_stats["errors"] += 1
            await self._notify_handlers("on_error", {"type": "decision_error", "error": str(e)})
            return False

    async def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary across all exchanges"""
        if not self.trading_orchestrator:
            return {}

        try:
            account_info = await self.trading_orchestrator.get_account_info()
            positions = await self.trading_orchestrator.get_positions()

            return {
                "account_info": account_info,
                "positions": {k: v.__dict__ for k, v in positions.items()},
                "total_positions": len(positions),
                "exchanges": self.system_config.exchanges,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting account summary: {e}")
            return {"error": str(e)}

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            orchestrator_stats = {}
            if self.trading_orchestrator:
                orchestrator_stats = await self.trading_orchestrator.get_statistics()

            return {
                "system_running": self._running,
                "system_initialized": self._initialized,
                "uptime_seconds": self.system_stats["uptime"],
                "total_signals_processed": self.system_stats["total_signals_processed"],
                "total_trades_executed": self.system_stats["total_trades_executed"],
                "total_exchanges_connected": self.system_stats["total_exchanges_connected"],
                "errors": self.system_stats["errors"],
                "warnings": self.system_stats["warnings"],
                "orchestrator_stats": orchestrator_stats,
                "config": self.system_config.trading_config.to_dict(),
                "exchanges": self.system_config.exchanges,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}

    async def emergency_stop(self) -> None:
        """Emergency stop all trading operations"""
        self.logger.warning("Emergency stop initiated!")

        try:
            if self.trading_orchestrator:
                await self.trading_orchestrator.emergency_stop()

            self.logger.info("Emergency stop completed")

        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")

    async def pause_trading(self) -> None:
        """Pause all trading operations"""
        if self.trading_orchestrator:
            await self.trading_orchestrator.pause_trading()

    async def resume_trading(self) -> None:
        """Resume all trading operations"""
        if self.trading_orchestrator:
            await self.trading_orchestrator.resume_trading()

    async def _initialize_trading_receiver(self) -> None:
        """Initialize the trading receiver"""
        try:
            # Create configuration for trading receiver
            receiver_config = {
                "exchanges": self._get_exchange_configs(),
                "enable_websockets": self.system_config.enable_websockets,
                "enable_paper_trading": self.system_config.enable_paper_trading,
                **self.system_config.custom_settings
            }

            # Create trading receiver
            self.trading_receiver = TradingReceiver(receiver_config)

            self.logger.info("Trading receiver initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize trading receiver: {e}")
            raise

    async def _initialize_trading_orchestrator(self) -> None:
        """Initialize the trading orchestrator"""
        if not self.trading_receiver:
            raise RuntimeError("Trading receiver must be initialized first")

        try:
            self.trading_orchestrator = TradingOrchestrator(
                self.system_config.trading_config,
                self.trading_receiver
            )

            self.logger.info("Trading orchestrator initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize trading orchestrator: {e}")
            raise

    async def _configure_exchanges(self) -> None:
        """Configure exchanges for the trading system"""
        try:
            exchange_configs = self._get_exchange_configs()

            for exchange_name in self.system_config.exchanges:
                config = exchange_configs.get(exchange_name, {})

                # Create exchange instance
                exchange = ExchangeFactory.get_exchange(exchange_name)

                # Register with trading receiver
                if self.trading_receiver:
                    success = await self.trading_receiver.exchange_registry.register_exchange(
                        exchange_name, exchange
                    )

                    if success:
                        self.system_stats["total_exchanges_connected"] += 1
                        self.logger.info(f"Exchange {exchange_name} configured successfully")
                    else:
                        self.logger.error(f"Failed to configure exchange {exchange_name}")

        except Exception as e:
            self.logger.error(f"Error configuring exchanges: {e}")
            raise

    def _get_exchange_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get exchange configurations"""
        # This would typically read from a configuration file or environment
        # For now, return a basic configuration
        configs = {}

        for exchange_name in self.system_config.exchanges:
            configs[exchange_name] = {
                "api_key": "",  # Would be loaded from secure config
                "api_secret": "",  # Would be loaded from secure config
                "sandbox": self.system_config.enable_paper_trading,
                "rate_limit": 1200,
                "timeout": 30
            }

        return configs

    async def _notify_handlers(self, event_type: str, data: Any) -> None:
        """Notify registered handlers"""
        if event_type in self.system_handlers:
            for handler in self.system_handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    self.logger.error(f"Error in system handler: {e}")


# Factory function to create a unified trading system
async def create_trading_system(
    trading_config: Optional[TradingConfig] = None,
    exchanges: Optional[List[str]] = None,
    enable_websockets: bool = True,
    enable_paper_trading: bool = True
) -> UnifiedTradingSystem:
    """Create and initialize a unified trading system"""

    if trading_config is None:
        trading_config = TradingConfig()

    system_config = SystemConfig(
        trading_config=trading_config,
        exchanges=exchanges or ["binance"],
        enable_websockets=enable_websockets,
        enable_paper_trading=enable_paper_trading
    )

    system = UnifiedTradingSystem(system_config)
    await system.initialize()

    return system


# Example usage function
async def example_trading_system():
    """Example of how to use the unified trading system"""
    try:
        # Create system configuration
        trading_config = TradingConfig(
            mode=TradingMode.PAPER,  # Use paper trading for testing
            symbols=["BTCUSDT", "ETHUSDT"],
            max_position_size=1000.0,
            max_daily_loss=50.0
        )

        # Create trading system
        system = await create_trading_system(
            trading_config=trading_config,
            exchanges=["binance", "okx"],
            enable_paper_trading=True
        )

        # Register event handlers
        async def on_trade_executed(data):
            print(f"Trade executed: {data}")

        system.register_handler("on_trading_start", on_trade_executed)

        # Start the system
        await system.start()

        # Submit a trading signal
        signal = TradingSignal(
            symbol="BTCUSDT",
            action="buy",
            quantity=0.001,
            confidence=0.8,
            strategy="example_strategy"
        )

        success = await system.submit_trading_signal(signal)
        print(f"Signal submitted: {success}")

        # Get system status
        status = await system.get_system_status()
        print(f"System status: {status}")

        # Stop the system
        await system.stop()

    except Exception as e:
        print(f"Error in example: {e}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_trading_system())