# src/core/dependency_injection.py

import asyncio
from typing import Dict, Any, Optional, Type, TypeVar
from dataclasses import dataclass
from abc import ABC

from src.interfaces import (
    IAnalyst, IStrategist, ITactician, ISupervisor,
    IExchangeClient, IStateManager, IPerformanceReporter,
    IModelManager, IEventBus, EventBus
)
from src.utils.logger import system_logger

T = TypeVar('T')

@dataclass
class ServiceRegistration:
    """Service registration information"""
    interface: Type
    implementation: Type
    singleton: bool = True
    dependencies: Dict[str, str] = None
    instance: Any = None

class DependencyContainer:
    """
    Dependency injection container for managing component dependencies.
    Provides singleton and transient service registration and resolution.
    """
    
    def __init__(self):
        self.logger = system_logger.getChild('DependencyContainer')
        self._services: Dict[str, ServiceRegistration] = {}
        self._singleton_instances: Dict[str, Any] = {}
        
    def register_singleton(self, interface: Type[T], implementation: Type[T], 
                          dependencies: Optional[Dict[str, str]] = None) -> None:
        """
        Register a singleton service
        
        Args:
            interface: Interface type
            implementation: Implementation type
            dependencies: Optional dependency mapping
        """
        service_name = interface.__name__
        self._services[service_name] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            singleton=True,
            dependencies=dependencies or {}
        )
        self.logger.debug(f"Registered singleton: {service_name}")
        
    def register_transient(self, interface: Type[T], implementation: Type[T],
                          dependencies: Optional[Dict[str, str]] = None) -> None:
        """
        Register a transient service
        
        Args:
            interface: Interface type
            implementation: Implementation type
            dependencies: Optional dependency mapping
        """
        service_name = interface.__name__
        self._services[service_name] = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            singleton=False,
            dependencies=dependencies or {}
        )
        self.logger.debug(f"Registered transient: {service_name}")
        
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """
        Register an existing instance
        
        Args:
            interface: Interface type
            instance: Existing instance
        """
        service_name = interface.__name__
        self._services[service_name] = ServiceRegistration(
            interface=interface,
            implementation=type(instance),
            singleton=True,
            instance=instance
        )
        self._singleton_instances[service_name] = instance
        self.logger.debug(f"Registered instance: {service_name}")
        
    def resolve(self, interface: Type[T]) -> T:
        """
        Resolve a service instance
        
        Args:
            interface: Interface type to resolve
            
        Returns:
            Service instance
        """
        service_name = interface.__name__
        
        if service_name not in self._services:
            raise ValueError(f"Service not registered: {service_name}")
            
        registration = self._services[service_name]
        
        # Return existing singleton instance if available
        if registration.singleton and service_name in self._singleton_instances:
            return self._singleton_instances[service_name]
            
        # Return existing instance if available
        if registration.instance is not None:
            return registration.instance
            
        # Create new instance
        instance = self._create_instance(registration)
        
        # Store singleton instance
        if registration.singleton:
            self._singleton_instances[service_name] = instance
            
        return instance
        
    def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Create a new instance with dependencies"""
        implementation = registration.implementation
        
        # Get constructor parameters
        import inspect
        sig = inspect.signature(implementation.__init__)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            # Check if parameter is a registered service
            if param_name in registration.dependencies:
                dependency_interface = registration.dependencies[param_name]
                dependency_type = self._get_type_by_name(dependency_interface)
                params[param_name] = self.resolve(dependency_type)
            elif param.annotation != inspect.Parameter.empty:
                # Try to resolve by type annotation
                try:
                    params[param_name] = self.resolve(param.annotation)
                except ValueError:
                    # Parameter not registered, use default
                    if param.default != inspect.Parameter.empty:
                        params[param_name] = param.default
                    else:
                        self.logger.warning(f"Unresolved dependency: {param_name}")
                        
        return implementation(**params)
        
    def _get_type_by_name(self, type_name: str) -> Type:
        """Get type by name"""
        for registration in self._services.values():
            if registration.interface.__name__ == type_name:
                return registration.interface
        raise ValueError(f"Type not found: {type_name}")
        
    def get_registered_services(self) -> Dict[str, ServiceRegistration]:
        """Get all registered services"""
        return self._services.copy()
        
    def clear(self):
        """Clear all registrations and instances"""
        self._services.clear()
        self._singleton_instances.clear()
        self.logger.info("Dependency container cleared")

class ComponentFactory:
    """
    Factory for creating trading components with proper dependency injection.
    """
    
    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild('ComponentFactory')
        
    def create_analyst(self, exchange_client: IExchangeClient, 
                      state_manager: IStateManager) -> IAnalyst:
        """Create analyst instance"""
        # Register dependencies
        self.container.register_instance(IExchangeClient, exchange_client)
        self.container.register_instance(IStateManager, state_manager)
        
        return self.container.resolve(IAnalyst)
        
    def create_strategist(self, exchange_client: IExchangeClient,
                         state_manager: IStateManager) -> IStrategist:
        """Create strategist instance"""
        # Register dependencies
        self.container.register_instance(IExchangeClient, exchange_client)
        self.container.register_instance(IStateManager, state_manager)
        
        return self.container.resolve(IStrategist)
        
    def create_tactician(self, exchange_client: IExchangeClient,
                        state_manager: IStateManager,
                        performance_reporter: IPerformanceReporter) -> ITactician:
        """Create tactician instance"""
        # Register dependencies
        self.container.register_instance(IExchangeClient, exchange_client)
        self.container.register_instance(IStateManager, state_manager)
        self.container.register_instance(IPerformanceReporter, performance_reporter)
        
        return self.container.resolve(ITactician)
        
    def create_supervisor(self, exchange_client: IExchangeClient,
                         state_manager: IStateManager,
                         event_bus: IEventBus) -> ISupervisor:
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
        self.components: Dict[str, Any] = {}
        self.logger = system_logger.getChild('ModularTradingSystem')
        
    async def initialize(self, exchange_client: IExchangeClient,
                        state_manager: IStateManager,
                        performance_reporter: IPerformanceReporter):
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
        self.components['analyst'] = self.factory.create_analyst(exchange_client, state_manager)
        self.components['strategist'] = self.factory.create_strategist(exchange_client, state_manager)
        self.components['tactician'] = self.factory.create_tactician(
            exchange_client, state_manager, performance_reporter
        )
        self.components['supervisor'] = self.factory.create_supervisor(
            exchange_client, state_manager, self.event_bus
        )
        
        # Set up event subscriptions
        await self._setup_event_subscriptions()
        
        self.logger.info("Modular trading system initialized")
        
    async def _setup_event_subscriptions(self):
        """Set up event subscriptions between components"""
        # Analyst publishes analysis results
        await self.event_bus.subscribe(
            EventType.MARKET_DATA_RECEIVED,
            self.components['analyst'].analyze_market_data
        )
        
        # Strategist subscribes to analysis results
        await self.event_bus.subscribe(
            EventType.ANALYSIS_COMPLETED,
            self.components['strategist'].formulate_strategy
        )
        
        # Tactician subscribes to strategy results
        await self.event_bus.subscribe(
            EventType.STRATEGY_FORMULATED,
            self.components['tactician'].execute_trade_decision
        )
        
        # Supervisor monitors all events
        await self.event_bus.subscribe(
            EventType.TRADE_EXECUTED,
            self.components['supervisor'].monitor_performance
        )
        
    async def start(self):
        """Start all components"""
        self.logger.info("Starting modular trading system")
        
        tasks = []
        for name, component in self.components.items():
            if hasattr(component, 'start'):
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
            if hasattr(component, 'stop'):
                await component.stop()
                
        self.logger.info("Modular trading system stopped")
        
    def get_component(self, name: str) -> Any:
        """Get a component by name"""
        return self.components.get(name)
        
    def get_all_components(self) -> Dict[str, Any]:
        """Get all components"""
        return self.components.copy() 