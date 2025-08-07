# src/core/enhanced_factories.py

"""
Enhanced factory classes that use dependency injection.

This module provides factory classes that create trading components
using proper dependency injection patterns.
"""

from typing import Any

from src.core.dependency_injection import DependencyContainer
from src.interfaces.base_interfaces import (
    IAnalyst,
    IExchangeClient,
    IPerformanceReporter,
    IStateManager,
    IStrategist,
    ISupervisor,
    ITactician,
)
from src.utils.logger import system_logger


class TradingSystemFactory:
    """
    Factory for creating complete trading systems with dependency injection.
    """

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("TradingSystemFactory")

    async def create_complete_trading_system(
        self,
        exchange_client: IExchangeClient,
        state_manager: IStateManager,
        performance_reporter: IPerformanceReporter,
    ) -> dict[str, Any]:
        """
        Create a complete trading system with all components.
        
        Args:
            exchange_client: Exchange client instance
            state_manager: State manager instance
            performance_reporter: Performance reporter instance
            
        Returns:
            Dictionary containing all trading components
        """
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
                        raise RuntimeError(f"Failed to initialize {name}")
            
            self.logger.info("Complete trading system created successfully")
            return components
            
        except Exception as e:
            self.logger.error(f"Failed to create trading system: {e}")
            raise


class ExchangeClientFactory:
    """
    Factory for creating exchange clients with dependency injection support.
    """

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("ExchangeClientFactory")

    def create_exchange_client(self, exchange_name: str, config: dict[str, Any]) -> IExchangeClient:
        """
        Create exchange client based on configuration.
        
        Args:
            exchange_name: Name of the exchange (binance, mexc, etc.)
            config: Exchange configuration
            
        Returns:
            Exchange client instance
        """
        try:
            from exchange.factory import ExchangeFactory
            
            factory = ExchangeFactory()
            client = factory.create_exchange(exchange_name, config)
            
            # Register the client in the container
            self.container.register_instance(IExchangeClient, client)
            
            self.logger.info(f"Created {exchange_name} exchange client")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to create exchange client: {e}")
            raise


class StateManagerFactory:
    """
    Factory for creating state managers with dependency injection support.
    """

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("StateManagerFactory")

    def create_state_manager(self, config: dict[str, Any]) -> IStateManager:
        """
        Create state manager based on configuration.
        
        Args:
            config: State manager configuration
            
        Returns:
            State manager instance
        """
        try:
            from src.utils.state_manager import StateManager
            
            state_manager = StateManager(config)
            
            # Register the state manager in the container
            self.container.register_instance(IStateManager, state_manager)
            
            self.logger.info("Created state manager")
            return state_manager
            
        except Exception as e:
            self.logger.error(f"Failed to create state manager: {e}")
            raise


class PerformanceReporterFactory:
    """
    Factory for creating performance reporters with dependency injection support.
    """

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("PerformanceReporterFactory")

    def create_performance_reporter(
        self, 
        config: dict[str, Any],
        db_manager: Any = None
    ) -> IPerformanceReporter:
        """
        Create performance reporter based on configuration.
        
        Args:
            config: Performance reporter configuration
            db_manager: Optional database manager
            
        Returns:
            Performance reporter instance
        """
        try:
            from src.supervisor.performance_reporter import PerformanceReporter
            
            if db_manager:
                reporter = PerformanceReporter(config, db_manager)
            else:
                reporter = PerformanceReporter(config)
            
            # Register the reporter in the container
            self.container.register_instance(IPerformanceReporter, reporter)
            
            self.logger.info("Created performance reporter")
            return reporter
            
        except Exception as e:
            self.logger.error(f"Failed to create performance reporter: {e}")
            raise


class DatabaseManagerFactory:
    """
    Factory for creating database managers with dependency injection support.
    """

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("DatabaseManagerFactory")

    def create_database_manager(self, config: dict[str, Any]) -> Any:
        """
        Create database manager based on configuration.
        
        Args:
            config: Database configuration
            
        Returns:
            Database manager instance
        """
        try:
            db_config = config.get("database", {})
            db_type = db_config.get("type", "influxdb")
            
            if db_type == "influxdb":
                from src.database.influxdb_manager import InfluxDBManager
                db_manager = InfluxDBManager(db_config)
            elif db_type == "firestore":
                from src.database.firestore_manager import FirestoreManager
                db_manager = FirestoreManager(db_config)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            self.logger.info(f"Created {db_type} database manager")
            return db_manager
            
        except Exception as e:
            self.logger.error(f"Failed to create database manager: {e}")
            raise


class ComprehensiveFactory:
    """
    Comprehensive factory that creates all system components with proper dependency injection.
    """

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("ComprehensiveFactory")
        
        # Initialize sub-factories
        self.trading_factory = TradingSystemFactory(container)
        self.exchange_factory = ExchangeClientFactory(container)
        self.state_factory = StateManagerFactory(container)
        self.performance_factory = PerformanceReporterFactory(container)
        self.database_factory = DatabaseManagerFactory(container)

    async def create_full_system(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Create a complete trading system with all dependencies.
        
        Args:
            config: System configuration
            
        Returns:
            Dictionary containing all system components
        """
        try:
            self.logger.info("Creating full trading system")
            
            # Create infrastructure components
            db_manager = self.database_factory.create_database_manager(config)
            
            # Create core dependencies
            exchange_config = config.get("exchange", {})
            exchange_name = exchange_config.get("name", "binance")
            exchange_client = self.exchange_factory.create_exchange_client(
                exchange_name, exchange_config
            )
            
            state_manager = self.state_factory.create_state_manager(
                config.get("state_manager", {})
            )
            
            performance_reporter = self.performance_factory.create_performance_reporter(
                config.get("performance_reporter", {}), db_manager
            )
            
            # Create trading components
            trading_components = await self.trading_factory.create_complete_trading_system(
                exchange_client, state_manager, performance_reporter
            )
            
            # Combine all components
            full_system = {
                "db_manager": db_manager,
                "exchange_client": exchange_client,
                "state_manager": state_manager,
                "performance_reporter": performance_reporter,
                **trading_components,
            }
            
            self.logger.info("Full trading system created successfully")
            return full_system
            
        except Exception as e:
            self.logger.error(f"Failed to create full system: {e}")
            raise

    def get_container_info(self) -> dict[str, Any]:
        """Get information about the dependency injection container."""
        return {
            "registered_services": list(self.container.get_all_services().keys()),
            "container_type": type(self.container).__name__,
        }