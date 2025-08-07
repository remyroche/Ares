# src/core/service_registry.py

"""
Service registry for dependency injection container configuration.

This module provides centralized service registration for all trading components,
ensuring proper dependency injection throughout the system.
"""

from typing import Any

from src.analyst.analyst import Analyst
from src.components.modular_analyst import ModularAnalyst
from src.components.modular_strategist import ModularStrategist
from src.components.modular_tactician import ModularTactician
from src.core.dependency_injection import DependencyContainer, ServiceLifetime
from src.interfaces.base_interfaces import (
    IAnalyst,
    IEventBus,
    IExchangeClient,
    IPerformanceReporter,
    IStateManager,
    IStrategist,
    ISupervisor,
    ITactician,
)
from src.interfaces.event_bus import EventBus
from src.strategist.strategist import Strategist
from src.supervisor.supervisor import Supervisor
from src.tactician.tactician import Tactician
from src.utils.logger import system_logger


class ServiceRegistry:
    """
    Centralized service registry for dependency injection configuration.
    """

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("ServiceRegistry")

    def register_all_services(self, config: dict[str, Any]) -> None:
        """Register all trading system services."""
        self.logger.info("Registering all trading system services")
        
        # Register core infrastructure services
        self._register_core_services(config)
        
        # Register trading components
        self._register_trading_components(config)
        
        # Register specialized services
        self._register_specialized_services(config)
        
        self.logger.info("All services registered successfully")

    def _register_core_services(self, config: dict[str, Any]) -> None:
        """Register core infrastructure services."""
        # Event bus as singleton
        self.container.register(
            IEventBus,
            EventBus,
            lifetime=ServiceLifetime.SINGLETON,
            config=config.get("event_bus", {})
        )

    def _register_trading_components(self, config: dict[str, Any]) -> None:
        """Register trading component services."""
        # Determine which implementations to use based on config
        use_modular = config.get("use_modular_components", True)
        
        if use_modular:
            # Register modular implementations
            self.container.register(
                IAnalyst,
                ModularAnalyst,
                lifetime=ServiceLifetime.SINGLETON,
                config=config.get("analyst", {})
            )
            
            self.container.register(
                IStrategist,
                ModularStrategist,
                lifetime=ServiceLifetime.SINGLETON,
                config=config.get("strategist", {})
            )
            
            self.container.register(
                ITactician,
                ModularTactician,
                lifetime=ServiceLifetime.SINGLETON,
                config=config.get("tactician", {})
            )
        else:
            # Register standard implementations
            self.container.register(
                IAnalyst,
                Analyst,
                lifetime=ServiceLifetime.SINGLETON,
                config=config.get("analyst", {})
            )
            
            self.container.register(
                IStrategist,
                Strategist,
                lifetime=ServiceLifetime.SINGLETON,
                config=config.get("strategist", {})
            )
            
            self.container.register(
                ITactician,
                Tactician,
                lifetime=ServiceLifetime.SINGLETON,
                config=config.get("tactician", {})
            )

        # Supervisor (always use the enhanced version)
        self.container.register(
            ISupervisor,
            Supervisor,
            lifetime=ServiceLifetime.SINGLETON,
            config=config.get("supervisor", {})
        )

    def _register_specialized_services(self, config: dict[str, Any]) -> None:
        """Register specialized services."""
        # Register factories for complex service creation
        self._register_exchange_factories(config)
        self._register_training_services(config)

    def _register_exchange_factories(self, config: dict[str, Any]) -> None:
        """Register exchange client factories."""
        def exchange_factory(container: DependencyContainer) -> IExchangeClient:
            """Factory for creating exchange clients based on configuration."""
            from exchange.factory import ExchangeFactory
            
            exchange_config = config.get("exchange", {})
            exchange_name = exchange_config.get("name", "binance")
            
            factory = ExchangeFactory()
            return factory.create_exchange(exchange_name, exchange_config)

        self.container.register_factory(IExchangeClient, exchange_factory)

    def _register_training_services(self, config: dict[str, Any]) -> None:
        """Register training-related services."""
        from src.training.training_manager import TrainingManager
        
        def training_manager_factory(container: DependencyContainer) -> TrainingManager:
            """Factory for creating training manager."""
            return TrainingManager(config.get("training", {}))

        self.container.register_factory(TrainingManager, training_manager_factory)

    def register_runtime_services(
        self,
        exchange_client: IExchangeClient,
        state_manager: IStateManager,
        performance_reporter: IPerformanceReporter,
    ) -> None:
        """Register runtime services that are created externally."""
        self.container.register_instance(IExchangeClient, exchange_client)
        self.container.register_instance(IStateManager, state_manager)
        self.container.register_instance(IPerformanceReporter, performance_reporter)
        
        self.logger.info("Runtime services registered")

    def get_registered_services(self) -> list[str]:
        """Get list of all registered service names."""
        return [service_type.__name__ for service_type in self.container.get_all_services().keys()]


def create_configured_container(config: dict[str, Any]) -> DependencyContainer:
    """
    Create and configure a dependency injection container with all services.
    
    Args:
        config: System configuration dictionary
        
    Returns:
        Configured DependencyContainer instance
    """
    container = DependencyContainer(config)
    registry = ServiceRegistry(container)
    registry.register_all_services(config)
    
    return container