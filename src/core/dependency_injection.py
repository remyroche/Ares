# src/core/dependency_injection.py

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from src.interfaces import (
    EventBus,
    EventType,
    IAnalyst,
    IEventBus,
    IExchangeClient,
    IPerformanceReporter,
    IStateManager,
    IStrategist,
    ISupervisor,
    ITactician,
)
from src.utils.logger import system_logger

T = TypeVar("T")


@dataclass
class ServiceRegistration:
    """Enhanced service registration with configuration support."""

    service_type: type
    implementation: type | None = None
    singleton: bool = True
    config: dict[str, Any] | None = None
    factory_method: str | None = None
    dependencies: dict[str, str] | None = None


class DependencyContainer:
    """
    Enhanced dependency injection container with configuration management.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self._services: dict[str, ServiceRegistration] = {}
        self._instances: dict[str, Any] = {}
        self._config: dict[str, Any] = config or {}
        self._factories: dict[str, Callable] = {}
        self.logger = system_logger.getChild("DependencyContainer")

    def register(
        self,
        service_name: str,
        service_type: type,
        implementation: type | None = None,
        singleton: bool = True,
        config: dict[str, Any] | None = None,
        dependencies: dict[str, str] | None = None,
    ) -> None:
        """Register a service with enhanced configuration support."""
        self._services[service_name] = ServiceRegistration(
            service_type=service_type,
            implementation=implementation or service_type,
            singleton=singleton,
            config=config,
            dependencies=dependencies,
        )
        self.logger.debug(
            f"Registered service: {service_name} -> {service_type.__name__}",
        )

    def register_factory(self, service_name: str, factory_func: Callable) -> None:
        """Register a factory function for service creation."""
        self._factories[service_name] = factory_func
        self.logger.debug(f"Registered factory for: {service_name}")

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback."""
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value
        self.logger.debug(f"Set config: {key} = {value}")

    def get_service_config(self, service_name: str) -> dict[str, Any]:
        """Get service-specific configuration."""
        service = self._services.get(service_name)
        if service and service.config:
            return service.config
        return {}

    def resolve(self, service_name: str) -> Any:
        """Resolve a service with enhanced error handling."""
        try:
            # Check if instance already exists (for singletons)
            if service_name in self._instances:
                return self._instances[service_name]

            # Get service registration
            service_reg = self._services.get(service_name)
            if not service_reg:
                raise ValueError(f"Service '{service_name}' not registered")

            # Create instance
            instance = self._create_instance(service_name, service_reg)

            # Store instance if singleton
            if service_reg.singleton:
                self._instances[service_name] = instance

            return instance

        except Exception as e:
            self.logger.error(f"Failed to resolve service '{service_name}': {e}")
            raise

    def _create_instance(
        self,
        service_name: str,
        service_reg: ServiceRegistration,
    ) -> Any:
        """Create service instance with dependency injection."""
        try:
            # Use factory method if available
            if service_reg.factory_method and service_name in self._factories:
                factory_func = self._factories[service_name]
                return factory_func(self._config)

            # Get constructor parameters
            constructor_params = self._get_constructor_params(service_name, service_reg)

            # Create instance
            if constructor_params:
                instance = service_reg.implementation(**constructor_params)
            else:
                instance = service_reg.implementation()

            # Inject service-specific configuration if available
            if service_reg.config:
                self._inject_config(instance, service_reg.config)

            return instance

        except Exception as e:
            self.logger.error(f"Failed to create instance for '{service_name}': {e}")
            raise

    def _get_constructor_params(
        self,
        service_name: str,
        service_reg: ServiceRegistration,
    ) -> dict[str, Any]:
        """Get constructor parameters for service creation."""
        params = {}

        # Add service-specific config if available
        if service_reg.config:
            params["config"] = service_reg.config

        # Resolve dependencies if specified
        if service_reg.dependencies:
            for param_name, dep_service_name in service_reg.dependencies.items():
                try:
                    params[param_name] = self.resolve(dep_service_name)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to resolve dependency '{dep_service_name}' for '{param_name}': {e}",
                    )

        return params

    def _inject_config(self, instance: Any, config: dict[str, Any]) -> None:
        """Inject configuration into service instance."""
        try:
            if hasattr(instance, "config"):
                instance.config = config
            elif hasattr(instance, "_config"):
                instance._config = config
        except Exception as e:
            self.logger.warning(f"Failed to inject config into instance: {e}")

    def register_config_service(self, config: dict[str, Any]) -> None:
        """Register configuration as a service."""
        self._config.update(config)
        self.logger.info(f"Registered configuration with {len(config)} keys")

    def get_all_services(self) -> dict[str, ServiceRegistration]:
        """Get all registered services."""
        return self._services.copy()

    def clear(self) -> None:
        """Clear all registered services and instances."""
        self._services.clear()
        self._instances.clear()
        self._factories.clear()
        self.logger.info("Cleared all services and instances")


class ServiceLocator:
    """Service locator pattern implementation."""

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("ServiceLocator")

    def get_service(self, service_name: str) -> Any:
        """Get service from container."""
        return self.container.resolve(service_name)

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.container.get_config(key, default)


class AsyncServiceContainer(DependencyContainer):
    """Async-aware dependency container."""

    async def resolve_async(self, service_name: str) -> Any:
        """Resolve service asynchronously."""
        try:
            # Check if instance already exists
            if service_name in self._instances:
                return self._instances[service_name]

            # Get service registration
            service_reg = self._services.get(service_name)
            if not service_reg:
                raise ValueError(f"Service '{service_name}' not registered")

            # Create instance (potentially async)
            instance = await self._create_instance_async(service_name, service_reg)

            # Store instance if singleton
            if service_reg.singleton:
                self._instances[service_name] = instance

            return instance

        except Exception as e:
            self.logger.error(f"Failed to resolve async service '{service_name}': {e}")
            raise

    async def _create_instance_async(
        self,
        service_name: str,
        service_reg: ServiceRegistration,
    ) -> Any:
        """Create service instance asynchronously."""
        try:
            # Use factory method if available
            if service_reg.factory_method and service_name in self._factories:
                factory_func = self._factories[service_name]
                if asyncio.iscoroutinefunction(factory_func):
                    return await factory_func(self._config)
                return factory_func(self._config)

            # Get constructor parameters
            constructor_params = self._get_constructor_params(service_name, service_reg)

            # Create instance
            if constructor_params:
                instance = service_reg.implementation(**constructor_params)
            else:
                instance = service_reg.implementation()

            # Initialize async if needed
            if hasattr(instance, "initialize") and asyncio.iscoroutinefunction(
                instance.initialize,
            ):
                await instance.initialize()

            # Inject service-specific configuration
            if service_reg.config:
                self._inject_config(instance, service_reg.config)

            return instance

        except Exception as e:
            self.logger.error(
                f"Failed to create async instance for '{service_name}': {e}",
            )
            raise


class ComponentFactory:
    """
    Factory for creating trading components with proper dependency injection.
    """

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("ComponentFactory")

    def create_analyst(
        self,
        exchange_client: IExchangeClient,
        state_manager: IStateManager,
    ) -> IAnalyst:
        """Create analyst instance"""
        # Register dependencies
        self.container.register_instance(IExchangeClient, exchange_client)
        self.container.register_instance(IStateManager, state_manager)

        return self.container.resolve(IAnalyst)

    def create_strategist(
        self,
        exchange_client: IExchangeClient,
        state_manager: IStateManager,
    ) -> IStrategist:
        """Create strategist instance"""
        # Register dependencies
        self.container.register_instance(IExchangeClient, exchange_client)
        self.container.register_instance(IStateManager, state_manager)

        return self.container.resolve(IStrategist)

    def create_tactician(
        self,
        exchange_client: IExchangeClient,
        state_manager: IStateManager,
        performance_reporter: IPerformanceReporter,
    ) -> ITactician:
        """Create tactician instance"""
        # Register dependencies
        self.container.register_instance(IExchangeClient, exchange_client)
        self.container.register_instance(IStateManager, state_manager)
        self.container.register_instance(IPerformanceReporter, performance_reporter)

        return self.container.resolve(ITactician)

    def create_supervisor(
        self,
        exchange_client: IExchangeClient,
        state_manager: IStateManager,
        event_bus: IEventBus,
    ) -> ISupervisor:
        """Create supervisor instance"""
        # Register dependencies
        self.container.register_instance(IExchangeClient, exchange_client)
        self.container.register_instance(IStateManager, state_manager)
        self.container.register_instance(IEventBus, event_bus)

        return self.container.resolve(ISupervisor)


class ModularTradingSystem:
    """
    Modular trading system that uses dependency injection and event-driven architecture.
    """

    def __init__(self):
        self.container = DependencyContainer()
        self.factory = ComponentFactory(self.container)
        self.event_bus = EventBus()
        self.components: dict[str, Any] = {}
        self.logger = system_logger.getChild("ModularTradingSystem")

    async def initialize(
        self,
        exchange_client: IExchangeClient,
        state_manager: IStateManager,
        performance_reporter: IPerformanceReporter,
    ):
        """Initialize the modular trading system"""
        self.logger.info("Initializing modular trading system")

        # Register core services
        self.container.register_instance(IExchangeClient, exchange_client)
        self.container.register_instance(IStateManager, state_manager)
        self.container.register_instance(IPerformanceReporter, performance_reporter)
        self.container.register_instance(IEventBus, self.event_bus)

        # Start event bus
        await self.event_bus.start()

        # Create components
        self.components["analyst"] = self.factory.create_analyst(
            exchange_client,
            state_manager,
        )
        self.components["strategist"] = self.factory.create_strategist(
            exchange_client,
            state_manager,
        )
        self.components["tactician"] = self.factory.create_tactician(
            exchange_client,
            state_manager,
            performance_reporter,
        )
        self.components["supervisor"] = self.factory.create_supervisor(
            exchange_client,
            state_manager,
            self.event_bus,
        )

        # Set up event subscriptions
        await self._setup_event_subscriptions()

        self.logger.info("Modular trading system initialized")

    async def _setup_event_subscriptions(self):
        """Set up event subscriptions between components"""
        # Analyst publishes analysis results
        await self.event_bus.subscribe(
            EventType.MARKET_DATA_RECEIVED,
            self.components["analyst"].analyze_market_data,
        )

        # Strategist subscribes to analysis results
        await self.event_bus.subscribe(
            EventType.ANALYSIS_COMPLETED,
            self.components["strategist"].formulate_strategy,
        )

        # Tactician subscribes to strategy results
        await self.event_bus.subscribe(
            EventType.STRATEGY_FORMULATED,
            self.components["tactician"].execute_trade_decision,
        )

        # Supervisor monitors all events
        await self.event_bus.subscribe(
            EventType.TRADE_EXECUTED,
            self.components["supervisor"].monitor_performance,
        )

    async def start(self):
        """Start all components"""
        self.logger.info("Starting modular trading system")

        tasks = []
        for name, component in self.components.items():
            if hasattr(component, "start"):
                task = asyncio.create_task(component.start(), name=f"{name}_task")
                tasks.append(task)

        await asyncio.gather(*tasks)

    async def stop(self):
        """Stop all components"""
        self.logger.info("Stopping modular trading system")

        # Stop event bus
        await self.event_bus.stop()

        # Stop components
        for name, component in self.components.items():
            if hasattr(component, "stop"):
                await component.stop()

        self.logger.info("Modular trading system stopped")

    def get_component(self, name: str) -> Any:
        """Get a component by name"""
        return self.components.get(name)

    def get_all_components(self) -> dict[str, Any]:
        """Get all components"""
        return self.components.copy()
