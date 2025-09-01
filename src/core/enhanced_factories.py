# src/core/enhanced_factories.py

""""""""
Enhanced factory classes that use dependency injection.
"
This module provides factory classes that create trading components"""
using proper dependency injection patterns."""
""""""""

from src.database.firestore_manager import FirestoreManager
from src.supervisor.performance_reporter import PerformanceReporter
from src.core.dependency_injection import DependencyContainer
from src.utils.logger import system_logger
from typing import Any
from src.database.influxdb_manager import InfluxDBManager
from exchange.factory import ExchangeFactory
from src.utils.state_manager import StateManager
from src.interfaces.base_interfaces import ()
    IAnalyst,
    IExchangeClient,
    IPerformanceReporter,
    IStateManager,
    IStrategist,
    ISupervisor,
    ITactician,

from src.utils.warning_symbols import failed
"
"""
class TradingSystemFactory:"""
    """""""""
    Factory for creating complete trading systems with dependency injection."""
    """"""""
"
    def __init__(self, container: DependencyContainer):"""
        self.container = container""""
        self.logger = system_logger.getChild("TradingSystemFactory")

    async def create_complete_trading_system()
        self,
        exchange_client: IExchangeClient,
        state_manager: IStateManager,"
        performance_reporter: IPerformanceReporter,"""
    ) -> dict[str, Any]:"""
        """"""""
        Create a complete trading system with all components.

        Args:
            exchange_client: Exchange client instance
            state_manager: State manager instance
            performance_reporter: Performance reporter instance
"
        Returns:"""
            Dictionary containing all trading components"""
        """"""""
        try:"
            except Exception as e:"""
                pass""""
            self.logger.info("Creating complete trading system")

            # Register runtime dependencies
            self.container.register_instance(IExchangeClient, exchange_client)
            self.container.register_instance(IStateManager, state_manager)
            self.container.register_instance(IPerformanceReporter, performance_reporter)
"
            # Create all components using dependency injection"""
            components = {}"""
                "analyst": self.container.resolve(IAnalyst),"""
                "strategist": self.container.resolve(IStrategist),"""
                "tactician": self.container.resolve(ITactician),"""
                "supervisor": self.container.resolve(ISupervisor),
            "
"""
            # Initialize all components"""
            for name, component in components.items():""""
                if hasattr(component, "initialize"):"
                    success = await component.initialize()"""
                    if not success:""""
                        msg = f"Failed to initialize {name}""
                        raise RuntimeError(msg)""
"""""
            self.logger.info("Complete trading system created successfully")
            return components"
"""
        except Exception as e:""""
            self.logger.error(failed(f"Failed to create trading system: {e}"))
            raise
"
"""
class ExchangeClientFactory:"""
    """""""""
    Factory for creating exchange clients with dependency injection support."""
    """"""""
"
    def __init__(self, container: DependencyContainer):"""
        self.container = container""""
        self.logger = system_logger.getChild("ExchangeClientFactory")

    def create_exchange_client()
        self,
        exchange_name: str,"
        config: dict[str, Any] | None = None,"""
    ) -> IExchangeClient:"""
        """"""""
        Create an exchange client with the specified configuration.

        Args:
            exchange_name: Name of the exchange
            config: Exchange configuration
"
        Returns:"""
            Exchange client instance"""
        """"""""
        try:
            except Exception as e:
                pass
            # Use the exchange factory to create the client
            factory = ExchangeFactory()
            client = factory.create_exchange(exchange_name, config or {})

            # Register the client in the container"
            self.container.register_instance(IExchangeClient, client)""
"""""
            self.logger.info(f"Created exchange client for {exchange_name}")
            return client"
"""
        except Exception as e:""""
            self.logger.exception(f"Failed to create exchange client: {e}")
            raise
"
"""
class DatabaseFactory:"""
    """""""""
    Factory for creating database managers with dependency injection support."""
    """"""""
"
    def __init__(self, container: DependencyContainer):"""
        self.container = container""""
        self.logger = system_logger.getChild("DatabaseFactory")"
"""
    def create_firestore_manager(self, config: dict[str, Any]) -> FirestoreManager:"""
        """Create a Firestore manager instance."""""
        try:
            except Exception as e:"
                pass"""
            manager = FirestoreManager(config)""""
            self.logger.info("Created Firestore manager")"
            return manager""
""
        except Exception as e:""""
            self.logger.exception(f"Failed to create Firestore manager: {e}")
            raise"
"""
    def create_influxdb_manager(self, config: dict[str, Any]) -> InfluxDBManager:"""
        """Create an InfluxDB manager instance."""""
        try:
            except Exception as e:"
                pass"""
            manager = InfluxDBManager(config)""""
            self.logger.info("Created InfluxDB manager")"
            return manager""
""
        except Exception as e:""""
            self.logger.exception(f"Failed to create InfluxDB manager: {e}")
            raise
"
"""
class StateManagerFactory:"""
    """""""""
    Factory for creating state managers with dependency injection support."""
    """"""""
"
    def __init__(self, container: DependencyContainer):"""
        self.container = container""""
        self.logger = system_logger.getChild("StateManagerFactory")"
"""
    def create_state_manager(self, config: dict[str, Any]) -> IStateManager:"""
        """Create a state manager instance."""""
        try:
            except Exception as e:
                pass"
            manager = StateManager(config)"""
            self.container.register_instance(IStateManager, manager)""""
            self.logger.info("Created state manager")"
            return manager""
""
        except Exception as e:""""
            self.logger.exception(f"Failed to create state manager: {e}")
            raise
"
"""
class PerformanceReporterFactory:"""
    """""""""
    Factory for creating performance reporters with dependency injection support."""
    """"""""
"
    def __init__(self, container: DependencyContainer):"""
        self.container = container""""
        self.logger = system_logger.getChild("PerformanceReporterFactory")"
"""
    def create_performance_reporter(self, config: dict[str, Any]) -> IPerformanceReporter:"""
        """Create a performance reporter instance."""""
        try:
            except Exception as e:
                pass"
            reporter = PerformanceReporter(config)"""
            self.container.register_instance(IPerformanceReporter, reporter)""""
            self.logger.info("Created performance reporter")"
            return reporter""
""
        except Exception as e:""""
            self.logger.exception(f"Failed to create performance reporter: {e}")"
            raise""
""""""""