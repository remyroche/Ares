# src/core/enhanced_factories.py

from typing import Any, Dict
from src.core.dependency_injection import DependencyContainer
from src.interfaces.base_interfaces import (
    IAnalyst,
    IStrategist,
    ITactician,
    ISupervisor,
    IExchangeClient,
    IStateManager,
    IPerformanceReporter,
)


class TradingSystemFactory:
    """
    Factory for creating complete trading systems with dependency injection.
    """
    
    def __init__(self, container: DependencyContainer):
        self.container = container

    async def create_complete_trading_system(
        self,
        exchange_client: IExchangeClient,
        state_manager: IStateManager,
        performance_reporter: IPerformanceReporter,
    ) -> Dict[str, Any]:
        """Create a complete trading system."""
        try:
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

            return components

        except Exception as e:
            raise RuntimeError(f"Failed to create trading system: {e}")


class ExchangeClientFactory:
    """
    Factory for creating exchange clients with dependency injection support.
    """
    
    def __init__(self, container: DependencyContainer):
        self.container = container

    async def create_exchange_client(self, config: Dict[str, Any]) -> IExchangeClient:
        """Create an exchange client."""
        try:
            # This would create a specific exchange client implementation
            # For now, return a mock or raise NotImplementedError
            raise NotImplementedError("Exchange client creation not implemented")
        except Exception as e:
            raise RuntimeError(f"Failed to create exchange client: {e}")


class DatabaseFactory:
    """
    Factory for creating database connections with dependency injection support.
    """
    
    def __init__(self, container: DependencyContainer):
        self.container = container

    async def create_database_connection(self, config: Dict[str, Any]) -> Any:
        """Create a database connection."""
        try:
            # This would create a specific database connection
            # For now, return a mock or raise NotImplementedError
            raise NotImplementedError("Database connection creation not implemented")
        except Exception as e:
            raise RuntimeError(f"Failed to create database connection: {e}")


class ModelFactory:
    """
    Factory for creating ML models with dependency injection support.
    """
    
    def __init__(self, container: DependencyContainer):
        self.container = container

    async def create_model(self, model_type: str, config: Dict[str, Any]) -> Any:
        """Create a model of the specified type."""
        try:
            # This would create a specific model implementation
            # For now, return a mock or raise NotImplementedError
            raise NotImplementedError(f"Model creation for type {model_type} not implemented")
        except Exception as e:
            raise RuntimeError(f"Failed to create model: {e}")


class StrategyFactory:
    """
    Factory for creating trading strategies with dependency injection support.
    """
    
    def __init__(self, container: DependencyContainer):
        self.container = container

    async def create_strategy(self, strategy_type: str, config: Dict[str, Any]) -> IStrategist:
        """Create a strategy of the specified type."""
        try:
            # This would create a specific strategy implementation
            # For now, return a mock or raise NotImplementedError
            raise NotImplementedError(f"Strategy creation for type {strategy_type} not implemented")
        except Exception as e:
            raise RuntimeError(f"Failed to create strategy: {e}")
