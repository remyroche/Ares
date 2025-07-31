# src/main_modular.py

import asyncio
import signal
import sys
from typing import Dict, Any

from src.core.dependency_injection import ModularTradingSystem
from src.interfaces import EventType
from src.interfaces.base_interfaces import (
    IExchangeClient,
    IStateManager,
    IPerformanceReporter,
)
from src.utils.logger import system_logger
from src.config import settings
from src.database.sqlite_manager import SQLiteManager
from src.exchange.binance import BinanceExchange
from src.utils.state_manager import StateManager
from src.supervisor.performance_reporter import PerformanceReporter
from src.utils.model_manager import ModelManager
from src.utils.error_handler import (
    handle_errors,
)


class ModularTradingBot:
    """
    Modular trading bot that uses dependency injection and event-driven architecture.
    This provides better separation of concerns, easier testing, and more flexible component management.
    """

    def __init__(self):
        self.logger = system_logger.getChild("ModularTradingBot")
        self.modular_system = ModularTradingSystem()
        self.running = False

        # Core dependencies
        self.exchange_client: IExchangeClient = None
        self.state_manager: IStateManager = None
        self.performance_reporter: IPerformanceReporter = None
        self.db_manager = None
        self.model_manager = None

    @handle_errors(
        exceptions=(Exception,), default_return=False, context="initialize_system"
    )
    async def initialize_system(self) -> bool:
        """Initialize the modular trading system"""
        self.logger.info("Initializing modular trading system")

        try:
            # Initialize database
            self.db_manager = SQLiteManager()
            await self.db_manager.initialize()

            # Initialize core dependencies
            await self._initialize_core_dependencies()

            # Initialize modular system
            await self.modular_system.initialize(
                exchange_client=self.exchange_client,
                state_manager=self.state_manager,
                performance_reporter=self.performance_reporter,
            )

            # Set up event handlers
            await self._setup_event_handlers()

            self.logger.info("Modular trading system initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}", exc_info=True)
            return False

    async def _initialize_core_dependencies(self):
        """Initialize core dependencies"""
        # Initialize exchange client
        if settings.trading_environment == "PAPER":
            from src.paper_trader import PaperTrader

            self.exchange_client = PaperTrader(initial_equity=settings.initial_equity)
        else:
            self.exchange_client = BinanceExchange()

        # Initialize state manager asynchronously
        self.state_manager = await StateManager.create()

        # Initialize model manager asynchronously
        self.model_manager = ModelManager()
        await self.model_manager.initialize()

        # Initialize performance reporter
        self.performance_reporter = PerformanceReporter(settings, self.db_manager)

        self.logger.info("Core dependencies initialized")

    async def _setup_event_handlers(self):
        """Set up event handlers for system monitoring"""
        # Subscribe to system events
        await self.modular_system.event_bus.subscribe(
            EventType.SYSTEM_ERROR, self._handle_system_error
        )

        await self.modular_system.event_bus.subscribe(
            EventType.RISK_ALERT, self._handle_risk_alert
        )

        await self.modular_system.event_bus.subscribe(
            EventType.PERFORMANCE_UPDATE, self._handle_performance_update
        )

    @handle_errors(exceptions=(Exception,), default_return=None, context="start_system")
    async def start_system(self) -> None:
        """Start the modular trading system"""
        self.logger.info("Starting modular trading system")
        self.running = True

        try:
            # Start the modular system
            await self.modular_system.start()

            # Start monitoring loop
            await self._run_monitoring_loop()

        except Exception as e:
            self.logger.error(f"Failed to start system: {e}", exc_info=True)
            await self.stop_system()

    async def _run_monitoring_loop(self):
        """Run the main monitoring loop"""
        self.logger.info("Starting monitoring loop")

        while self.running:
            try:
                # Monitor performance
                supervisor = self.modular_system.get_component("supervisor")
                if supervisor:
                    await supervisor.monitor_performance()
                    await supervisor.manage_risk()
                    await supervisor.coordinate_components()

                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                self.logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(10)  # Wait before retry

    @handle_errors(exceptions=(Exception,), default_return=None, context="stop_system")
    async def stop_system(self) -> None:
        """Stop the modular trading system"""
        self.logger.info("Stopping modular trading system")
        self.running = False

        try:
            # Stop the modular system
            await self.modular_system.stop()

            # Close database connection
            if self.db_manager:
                await self.db_manager.close()

            self.logger.info("Modular trading system stopped")

        except Exception as e:
            self.logger.error(f"Error stopping system: {e}", exc_info=True)

    async def _handle_system_error(self, event) -> None:
        """Handle system error events"""
        error_data = event.data
        self.logger.error(f"System error: {error_data}")

        # Log error to database
        if self.db_manager:
            await self.db_manager.log_system_error(error_data)

        # Send alert if configured
        await self._send_error_alert(error_data)

    async def _handle_risk_alert(self, event) -> None:
        """Handle risk alert events"""
        alert_data = event.data
        self.logger.warning(f"Risk alert: {alert_data}")

        # Log alert to database
        if self.db_manager:
            await self.db_manager.log_risk_alert(alert_data)

        # Send alert if configured
        await self._send_risk_alert(alert_data)

    async def _handle_performance_update(self, event) -> None:
        """Handle performance update events"""
        performance_data = event.data
        self.logger.debug(f"Performance update: {performance_data}")

        # Log performance to database
        if self.db_manager:
            await self.db_manager.log_performance_update(performance_data)

    async def _send_error_alert(self, error_data: Dict[str, Any]) -> None:
        """Send error alert"""
        try:
            # This would typically send an email or notification
            self.logger.info(f"Error alert sent: {error_data}")
        except Exception as e:
            self.logger.error(f"Failed to send error alert: {e}")

    async def _send_risk_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send risk alert"""
        try:
            # This would typically send an email or notification
            self.logger.info(f"Risk alert sent: {alert_data}")
        except Exception as e:
            self.logger.error(f"Failed to send risk alert: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "running": self.running,
            "components": self.modular_system.get_all_components(),
            "event_bus_subscribers": {
                event_type.value: self.modular_system.event_bus.get_subscriber_count(
                    event_type
                )
                for event_type in EventType
            },
        }


async def main():
    """Main entry point for the modular trading bot"""
    bot = ModularTradingBot()

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        system_logger.info(f"Received signal {signum}, shutting down gracefully...")
        asyncio.create_task(bot.stop_system())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize system
        if not await bot.initialize_system():
            system_logger.error("Failed to initialize system")
            sys.exit(1)

        # Start system
        await bot.start_system()

    except KeyboardInterrupt:
        system_logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        system_logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Ensure system is stopped
        await bot.stop_system()
        system_logger.info("Modular trading bot shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
