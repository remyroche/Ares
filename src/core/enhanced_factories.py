# src/core/enhanced_factories.py

"""
Enhanced factory classes that use dependency injection.

This module provides factory classes that create trading components
using proper dependency injection patterns.
"""

from src.database.firestore_manager import FirestoreManager
from src.supervisor.performance_reporter import PerformanceReporter
from src.core.dependency_injection import DependencyContainer
from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from typing import Any, Dict, Optional
from src.database.influxdb_manager import InfluxDBManager
from exchange.factory import ExchangeFactory
from src.utils.state_manager import StateManager
from src.interfaces.base_interfaces import (
    IAnalyst,
    IExchangeClient,
    IPerformanceReporter,
    IStateManager,
    IStrategist,
    ISupervisor,
    ITactician,
)
from src.utils.warning_symbols import failed


class TradingSystemFactory:
    """Factory for creating complete trading systems with dependency injection."""
    
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("TradingSystemFactory")
        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="tradingsystemfactory initialization",
    )
    async def initialize(self) -> bool:
        """Initialize TradingSystemFactory."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {self.__class__.__name__}: {e}")
            return False

    async def create_complete_trading_system(
        self,
        exchange_client: IExchangeClient,
        state_manager: IStateManager,
        performance_reporter: IPerformanceReporter,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a complete trading system with all components."""
        try:
            self.logger.info("Creating complete trading system")

            # Register runtime dependencies
            self.container.register_instance(IExchangeClient, exchange_client)
            self.container.register_instance(IStateManager, state_manager)
            self.container.register_instance(IPerformanceReporter, performance_reporter)

            # Create all components using dependency injection
            components = {
                "analyst": self.container.resolve(IAnalyst),
                "strategist": self.container.resolve(IStrategist),
                "tactician": self.container.resolve(ITactician),
                "supervisor": self.container.resolve(ISupervisor),
            }

            # Initialize all components
            for name, component in components.items():
                if hasattr(component, "initialize"):
                    success = await component.initialize()
                    if not success:
                        msg = f"Failed to initialize {name}"
                        raise RuntimeError(msg)

            self.logger.info("Complete trading system created successfully")
            return components

        except Exception as e:
            self.logger.error(failed(f"Failed to create trading system: {e}"))
            raise


class ExchangeClientFactory:
    """Factory for creating exchange clients with dependency injection support."""
    
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("ExchangeClientFactory")
        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="exchangeclientfactory initialization",
    )
    async def initialize(self) -> bool:
        """Initialize ExchangeClientFactory."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {self.__class__.__name__}: {e}")
            return False

    def create_exchange_client(
        self,
        exchange_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> IExchangeClient:
        """Create an exchange client using the exchange factory."""
        try:
            # Use the exchange factory to create the client
            factory = ExchangeFactory()
            client = factory.create_exchange(exchange_name, config or {})

            # Register the client in the container
            self.container.register_instance(IExchangeClient, client)

            self.logger.info(f"Created exchange client for {exchange_name}")
            return client

        except Exception as e:
            self.logger.exception(f"Failed to create exchange client: {e}")
            raise


class DatabaseFactory:
    """Factory for creating database managers with dependency injection support."""
    
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("DatabaseFactory")
        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="databasefactory initialization",
    )
    async def initialize(self) -> bool:
        """Initialize DatabaseFactory."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {self.__class__.__name__}: {e}")
            return False

    def create_firestore_manager(self, config: Optional[Dict[str, Any]] = None) -> FirestoreManager:
        """Create a Firestore manager instance."""
        try:
            manager = FirestoreManager(config)
            self.logger.info("Created Firestore manager")
            return manager

        except Exception as e:
            self.logger.exception(f"Failed to create Firestore manager: {e}")
            raise

    def create_influxdb_manager(self, config: Optional[Dict[str, Any]] = None) -> InfluxDBManager:
        """Create an InfluxDB manager instance."""
        try:
            manager = InfluxDBManager(config)
            self.logger.info("Created InfluxDB manager")
            return manager

        except Exception as e:
            self.logger.exception(f"Failed to create InfluxDB manager: {e}")
            raise


class StateManagerFactory:
    """Factory for creating state managers with dependency injection support."""
    
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("StateManagerFactory")
        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="statemanagerfactory initialization",
    )
    async def initialize(self) -> bool:
        """Initialize StateManagerFactory."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {self.__class__.__name__}: {e}")
            return False

    def create_state_manager(self, config: Optional[Dict[str, Any]] = None) -> IStateManager:
        """Create a state manager instance."""
        try:
            manager = StateManager(config)
            self.container.register_instance(IStateManager, manager)
            self.logger.info("Created state manager")
            return manager

        except Exception as e:
            self.logger.exception(f"Failed to create state manager: {e}")
            raise


class PerformanceReporterFactory:
    """Factory for creating performance reporters with dependency injection support."""
    
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("PerformanceReporterFactory")
        self.is_initialized = False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="performancereporterfactory initialization",
    )
    async def initialize(self) -> bool:
        """Initialize PerformanceReporterFactory."""
        try:
            class_name = self.__class__.__name__
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {self.__class__.__name__}: {e}")
            return False

    def create_performance_reporter(self, config: Optional[Dict[str, Any]] = None) -> IPerformanceReporter:
        """Create a performance reporter instance."""
        try:
            reporter = PerformanceReporter(config)
            self.container.register_instance(IPerformanceReporter, reporter)
            self.logger.info("Created performance reporter")
            return reporter

        except Exception as e:
            self.logger.exception(f"Failed to create performance reporter: {e}")
            raise
