from __future__ import annotations
# src/core/service_registry.py

"""
Service registry for dependency injection container configuration.

This module provides centralized service registration for all trading components, ensuring proper dependency injection throughout the system.
"""

from typing import Any

from exchange.factory import ExchangeFactory
from src.analyst.analyst import Analyst
from src.components.modular_analyst import ModularAnalyst
from src.components.modular_strategist import ModularStrategist
from src.components.modular_tactician import ModularTactician
from src.core.dependency_injection import DependencyContainer, ServiceLifetime
from src.interfaces.base_interfaces import (
    IAnalyst,
    IEventBus,
    IStrategist,
    ISupervisor,
    ITactician,
)
from src.interfaces.event_bus import EventBus
from src.strategist.strategist import Strategist
from src.supervisor.supervisor import Supervisor
from src.tactician.tactician import Tactician
from src.training.training_manager import TrainingManager
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
            config=config.get("event_bus", {}),
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
                config=config.get("analyst", {}),
            )

            self.container.register(
                IStrategist,
                ModularStrategist,
                lifetime=ServiceLifetime.SINGLETON,
                config=config.get("strategist", {}),
            )

            self.container.register(
                ITactician,
                ModularTactician,
                lifetime=ServiceLifetime.SINGLETON,
                config=config.get("tactician", {}),
            )
        else:
            # Register standard implementations
            self.container.register(
                IAnalyst,
                Analyst,
                lifetime=ServiceLifetime.SINGLETON,
                config=config.get("analyst", {}),
            )

            self.container.register(
                IStrategist,
                Strategist,
                lifetime=ServiceLifetime.SINGLETON,
                config=config.get("strategist", {}),
            )

            self.container.register(
                ITactician,
                Tactician,
                lifetime=ServiceLifetime.SINGLETON,
                config=config.get("tactician", {}),
            )

        # Register supervisor (same for both modes)
        self.container.register(
            ISupervisor,
            Supervisor,
            lifetime=ServiceLifetime.SINGLETON,
            config=config.get("supervisor", {}),
        )

    def _register_specialized_services(self, config: dict[str, Any]) -> None:
        """Register specialized services."""
        # Register training manager
        self.container.register(
            TrainingManager,
            TrainingManager,
            lifetime=ServiceLifetime.SINGLETON,
            config=config.get("training", {}),
        )

        # Register exchange factory
        self.container.register(
            ExchangeFactory,
            ExchangeFactory,
            lifetime=ServiceLifetime.SINGLETON,
            config=config.get("exchange", {}),
        )

    def get_registered_services(self) -> dict[str, Any]:
        """Get all registered services."""
        return self.container.get_all_services()

    def validate_registrations(self) -> bool:
        """Validate that all required services are registered."""
        required_services = [
            IEventBus,
            IAnalyst,
            IStrategist,
            ITactician,
            ISupervisor,
        ]

        missing_services = []
        for service in required_services:
            if service not in self.container.get_all_services():
                missing_services.append(service.__name__)

        if missing_services:
            self.logger.error(f"Missing required services: {missing_services}")
            return False

        self.logger.info("All required services are registered")
        return True
