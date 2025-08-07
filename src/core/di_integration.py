# src/core/di_integration.py

"""
Integration module for dependency injection patterns.

This module demonstrates how to integrate all dependency injection patterns
throughout the Ares trading system.
"""

import asyncio
from typing import Any

from src.config import CONFIG
from src.core.dependency_injection import AsyncServiceContainer, ServiceLifetime
from src.core.service_registry import ServiceRegistry
from src.core.enhanced_factories import ComprehensiveFactory
from src.core.di_launcher import DILauncher
from src.training.di_training_manager import DITrainingManager
from src.analyst.di_analyst import DIAnalyst
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
from src.utils.logger import system_logger


class DIIntegration:
    """
    Integration class that demonstrates proper dependency injection usage
    throughout the Ares trading system.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or CONFIG
        self.logger = system_logger.getChild("DIIntegration")
        
        # Initialize DI container
        self.container = AsyncServiceContainer(self.config)
        self.registry = ServiceRegistry(self.container)
        
        # Initialize factories
        self.factory = ComprehensiveFactory(self.container)
        
        # System state
        self.is_initialized = False
        self.system_components: dict[str, Any] = {}

    async def demonstrate_full_di_integration(self) -> dict[str, Any]:
        """
        Demonstrate complete dependency injection integration.
        
        This method shows how all DI patterns work together in practice.
        """
        try:
            self.logger.info("Demonstrating full dependency injection integration")
            
            # Step 1: Register all services
            await self._register_all_services()
            
            # Step 2: Create core infrastructure
            infrastructure = await self._create_infrastructure()
            
            # Step 3: Create trading components with DI
            trading_components = await self._create_trading_components()
            
            # Step 4: Create specialized services
            specialized_services = await self._create_specialized_services()
            
            # Step 5: Wire everything together
            complete_system = {
                **infrastructure,
                **trading_components,
                **specialized_services,
            }
            
            # Step 6: Initialize all components
            await self._initialize_all_components(complete_system)
            
            self.system_components = complete_system
            self.is_initialized = True
            
            self.logger.info("Full DI integration demonstration completed")
            return complete_system
            
        except Exception as e:
            self.logger.error(f"DI integration demonstration failed: {e}")
            raise

    async def _register_all_services(self) -> None:
        """Register all services in the DI container."""
        self.logger.info("Registering all services")
        
        # Register core infrastructure services
        self.registry.register_all_services(self.config)
        
        # Register custom implementations
        self.container.register(
            IAnalyst,
            DIAnalyst,
            lifetime=ServiceLifetime.SINGLETON,
            config=self.config.get("analyst", {})
        )
        
        # Register training manager
        self.container.register(
            DITrainingManager,
            DITrainingManager,
            lifetime=ServiceLifetime.SINGLETON,
            config=self.config.get("training", {})
        )
        
        self.logger.info("All services registered")

    async def _create_infrastructure(self) -> dict[str, Any]:
        """Create core infrastructure components."""
        self.logger.info("Creating core infrastructure")
        
        # Create database manager
        db_manager = self.factory.database_factory.create_database_manager(self.config)
        
        # Create exchange client
        exchange_config = self.config.get("exchange", {})
        exchange_name = exchange_config.get("name", "binance")
        exchange_client = self.factory.exchange_factory.create_exchange_client(
            exchange_name, exchange_config
        )
        
        # Create state manager
        state_manager = self.factory.state_factory.create_state_manager(
            self.config.get("state_manager", {})
        )
        
        # Create performance reporter
        performance_reporter = self.factory.performance_factory.create_performance_reporter(
            self.config.get("performance_reporter", {}), db_manager
        )
        
        infrastructure = {
            "db_manager": db_manager,
            "exchange_client": exchange_client,
            "state_manager": state_manager,
            "performance_reporter": performance_reporter,
        }
        
        self.logger.info("Core infrastructure created")
        return infrastructure

    async def _create_trading_components(self) -> dict[str, Any]:
        """Create trading components using DI."""
        self.logger.info("Creating trading components with DI")
        
        # Resolve components from DI container
        analyst = await self.container.resolve_async(IAnalyst)
        strategist = await self.container.resolve_async(IStrategist)
        tactician = await self.container.resolve_async(ITactician)
        supervisor = await self.container.resolve_async(ISupervisor)
        event_bus = self.container.resolve(IEventBus)
        
        trading_components = {
            "analyst": analyst,
            "strategist": strategist,
            "tactician": tactician,
            "supervisor": supervisor,
            "event_bus": event_bus,
        }
        
        self.logger.info("Trading components created")
        return trading_components

    async def _create_specialized_services(self) -> dict[str, Any]:
        """Create specialized services using DI."""
        self.logger.info("Creating specialized services")
        
        # Create training manager
        training_manager = await self.container.resolve_async(DITrainingManager)
        
        specialized_services = {
            "training_manager": training_manager,
            "container": self.container,
            "registry": self.registry,
            "factory": self.factory,
        }
        
        self.logger.info("Specialized services created")
        return specialized_services

    async def _initialize_all_components(self, components: dict[str, Any]) -> None:
        """Initialize all components in dependency order."""
        self.logger.info("Initializing all components")
        
        # Initialize in dependency order
        initialization_order = [
            "event_bus",
            "training_manager", 
            "analyst",
            "strategist",
            "tactician",
            "supervisor",
        ]
        
        for component_name in initialization_order:
            component = components.get(component_name)
            if component and hasattr(component, "initialize"):
                success = await component.initialize()
                if not success:
                    raise RuntimeError(f"Failed to initialize {component_name}")
                self.logger.debug(f"Initialized {component_name}")
        
        self.logger.info("All components initialized")

    async def demonstrate_scope_management(self) -> dict[str, Any]:
        """Demonstrate dependency injection scope management."""
        self.logger.info("Demonstrating scope management")
        
        try:
            # Begin a new scope
            self.container.begin_scope("trading_session")
            
            # Create scoped services
            self.container.register(
                "scoped_service",
                str,  # Simple example
                lifetime=ServiceLifetime.SCOPED,
            )
            
            # Resolve services within scope
            scoped_service1 = self.container.resolve("scoped_service")
            scoped_service2 = self.container.resolve("scoped_service")
            
            # Should be the same instance within scope
            same_instance = scoped_service1 is scoped_service2
            
            # End the scope
            self.container.end_scope("trading_session")
            
            result = {
                "scope_demonstration": "completed",
                "same_instance_within_scope": same_instance,
            }
            
            self.logger.info("Scope management demonstration completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Scope demonstration failed: {e}")
            raise

    async def demonstrate_configuration_injection(self) -> dict[str, Any]:
        """Demonstrate configuration injection patterns."""
        self.logger.info("Demonstrating configuration injection")
        
        try:
            # Show how configuration is injected into components
            analyst = self.system_components.get("analyst")
            if analyst and hasattr(analyst, "config"):
                analyst_config = analyst.config
            else:
                analyst_config = {}
            
            training_manager = self.system_components.get("training_manager")
            if training_manager and hasattr(training_manager, "config"):
                training_config = training_manager.config
            else:
                training_config = {}
            
            result = {
                "configuration_injection": "demonstrated",
                "analyst_config_keys": list(analyst_config.keys()),
                "training_config_keys": list(training_config.keys()),
                "container_config_keys": list(self.container.get_config("", {}).keys()),
            }
            
            self.logger.info("Configuration injection demonstration completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Configuration injection demonstration failed: {e}")
            raise

    async def demonstrate_factory_patterns(self) -> dict[str, Any]:
        """Demonstrate factory pattern integration with DI."""
        self.logger.info("Demonstrating factory patterns")
        
        try:
            # Show how factories work with DI
            container_info = self.factory.get_container_info()
            
            # Create a new launcher to show factory usage
            launcher = DILauncher(self.config)
            system_info = launcher.get_system_info()
            
            result = {
                "factory_patterns": "demonstrated", 
                "container_info": container_info,
                "launcher_system_info": system_info,
                "registered_services": list(self.container.get_all_services().keys()),
            }
            
            self.logger.info("Factory patterns demonstration completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Factory patterns demonstration failed: {e}")
            raise

    async def run_complete_demonstration(self) -> dict[str, Any]:
        """Run complete dependency injection demonstration."""
        try:
            self.logger.info("Starting complete DI demonstration")
            
            results = {}
            
            # Full integration
            results["full_integration"] = await self.demonstrate_full_di_integration()
            
            # Scope management
            results["scope_management"] = await self.demonstrate_scope_management()
            
            # Configuration injection
            results["configuration_injection"] = await self.demonstrate_configuration_injection()
            
            # Factory patterns
            results["factory_patterns"] = await self.demonstrate_factory_patterns()
            
            # Summary
            results["summary"] = {
                "total_components": len(self.system_components),
                "is_initialized": self.is_initialized,
                "demonstration_status": "completed_successfully",
            }
            
            self.logger.info("Complete DI demonstration finished successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Complete DI demonstration failed: {e}")
            raise

    def get_container_metrics(self) -> dict[str, Any]:
        """Get metrics about the DI container."""
        all_services = self.container.get_all_services()
        
        return {
            "total_registered_services": len(all_services),
            "service_types": [service_type.__name__ for service_type in all_services.keys()],
            "singleton_count": len([
                s for s in all_services.values() 
                if s.lifetime == ServiceLifetime.SINGLETON
            ]),
            "transient_count": len([
                s for s in all_services.values() 
                if s.lifetime == ServiceLifetime.TRANSIENT
            ]),
            "scoped_count": len([
                s for s in all_services.values() 
                if s.lifetime == ServiceLifetime.SCOPED
            ]),
        }


# Convenience function for quick demonstration
async def run_di_demonstration(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a complete dependency injection demonstration."""
    integration = DIIntegration(config)
    return await integration.run_complete_demonstration()