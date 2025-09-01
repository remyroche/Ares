# src/core/di_integration.py

"""
Integration module for dependency injection patterns.

This module demonstrates how to integrate all dependency injection patterns
throughout the Ares trading system.
"""

from src.core.dependency_injection import DependencyContainer, ServiceLifetime
from src.core.enhanced_factories import TradingSystemFactory
from src.core.service_registry import ServiceRegistry
from src.utils.logger import system_logger
from typing import Any
from src.analyst.di_analyst import DIAnalyst
from src.config import CONFIG
from src.training.di_training_manager import DITrainingManager
from src.interfaces.base_interfaces import (
import IAnalyst,
    IAnalyst,
    IEventBus,
    IStrategist,
    ISupervisor,
    ITactician,
)


class DIIntegration:
    """
    Integration class that demonstrates proper dependency injection usage
    throughout the Ares trading system.
    """

    def __init__(self, config: dict[str, Any] | None = None):
    pass
    pass
        self.config = config or CONFIG
        self.logger = system_logger.getChild("DIIntegration")

        # Initialize DI container
        self.container = DependencyContainer(self.config)
        self.registry = ServiceRegistry(self.container)

        # Initialize factories
        self.factory = TradingSystemFactory(self.container)

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

    except Exception as e:
        pass
    except Exception as e:
        pass
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
            self.logger.exception(f"DI integration demonstration failed: {e}")
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
            config=self.config.get("analyst", {}),
        )

        self.container.register(
            DITrainingManager,
            DITrainingManager,
            lifetime=ServiceLifetime.SINGLETON,
            config=self.config.get("training", {}),
        )

        self.logger.info("All services registered successfully")

    async def _create_infrastructure(self) -> dict[str, Any]:
        """Create core infrastructure components."""
        self.logger.info("Creating infrastructure components")

        # Create event bus
        event_bus = self.container.resolve(IEventBus)

        # Create other infrastructure components as needed
        infrastructure = {
            "event_bus": event_bus,
        }

        self.logger.info("Infrastructure components created")
        return infrastructure

    async def _create_trading_components(self) -> dict[str, Any]:
        """Create trading components using DI."""
        self.logger.info("Creating trading components")

        # Create trading components through DI
        components = {
            "analyst": self.container.resolve(IAnalyst),
            "strategist": self.container.resolve(IStrategist),
            "tactician": self.container.resolve(ITactician),
            "supervisor": self.container.resolve(ISupervisor),
        }

        self.logger.info("Trading components created")
        return components

    async def _create_specialized_services(self) -> dict[str, Any]:
        """Create specialized services."""
        self.logger.info("Creating specialized services")

        # Create training manager
        training_manager = self.container.resolve(DITrainingManager)

        specialized_services = {
            "training_manager": training_manager,
        }

        self.logger.info("Specialized services created")
        return specialized_services

    async def _initialize_all_components(self, components: dict[str, Any]) -> None:
        """Initialize all components."""
        self.logger.info("Initializing all components")

        for name, component in components.items():
    pass
    pass
            if hasattr(component, "initialize"):
    pass
    pass
                try:
                    success = await component.initialize()
    except Exception as e:
        pass
    except Exception as e:
        pass
                    if success:
    pass
    pass
                        self.logger.info(f"Initialized component: {name}")
                    else:
                        self.logger.error(f"Failed to initialize component: {name}")
                except Exception as e:
                    self.logger.exception(f"Error initializing {name}: {e}")

        self.logger.info("Component initialization completed")

    def get_integration_status(self) -> dict[str, Any]:
    pass
    pass
        """Get integration status."""
        return {
            "is_initialized": self.is_initialized,
            "components": list(self.system_components.keys()),
            "container_services": list(self.container.get_all_services().keys()),
        }

    async def shutdown(self) -> None:
        """Shutdown the integration."""
        self.logger.info("Shutting down DI integration")

        for name, component in self.system_components.items():
    pass
    pass
            if hasattr(component, "shutdown"):
    pass
    pass
                try:
                    await component.shutdown()
    except Exception as e:
        pass
    except Exception as e:
        pass
                    self.logger.info(f"Shutdown component: {name}")
                except Exception as e:
                    self.logger.exception(f"Error shutting down {name}: {e}")

        self.is_initialized = False
        self.logger.info("DI integration shutdown completed")


# Convenience function for quick integration demonstration
async def demonstrate_di_integration(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Quick demonstration of DI integration."""
    integration = DIIntegration(config)
    return await integration.demonstrate_full_di_integration()
