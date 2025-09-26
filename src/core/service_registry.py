"""Dependency injection service registration helpers."""

from __future__ import annotations

from typing import Any

from exchange.factory import ExchangeFactory
from src.analyst.analyst import Analyst
from src.components.modular_analyst import ModularAnalyst
from src.components.modular_strategist import ModularStrategist
from src.components.modular_tactician import ModularTactician
from src.interfaces.base_interfaces import (
    IAnalyst,
    IEventBus,
    IStrategist,
    ISupervisor,
    ITactician,
)
from src.interfaces.event_bus import EventBus
from src.strategist.strategist import Strategist
from src.supervisor.main import Supervisor
from src.tactician.tactician import Tactician
from src.training.core.training_manager import TrainingManager
from src.utils.logger import system_logger

from .dependency_injection import DependencyContainer, ServiceLifetime


class ServiceRegistry:
    """Centralised service registration for the dependency injection container."""

    def __init__(self, container: DependencyContainer) -> None:
        self.container = container
        self.logger = system_logger.getChild("ServiceRegistry")

    def register_all_services(self, config: dict[str, Any]) -> None:
        """Register the full set of trading system services."""

        self.logger.info("Registering all trading system services")
        self._register_core_services(config)
        self._register_trading_components(config)
        self._register_specialised_services(config)
        self.logger.info("All services registered successfully")

    def _register_core_services(self, config: dict[str, Any]) -> None:
        """Register shared infrastructure services."""

        self.container.register(
            IEventBus,
            EventBus,
            lifetime=ServiceLifetime.SINGLETON,
            config=config.get("event_bus", {}),
        )

    def _register_trading_components(self, config: dict[str, Any]) -> None:
        """Register trading system components with the container."""

        use_modular = config.get("use_modular_components", True)

        if use_modular:
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

        self.container.register(
            ISupervisor,
            Supervisor,
            lifetime=ServiceLifetime.SINGLETON,
            config=config.get("supervisor", {}),
        )

    def _register_specialised_services(self, config: dict[str, Any]) -> None:
        """Register specialised services that back the trading system."""

        self.container.register(
            TrainingManager,
            TrainingManager,
            lifetime=ServiceLifetime.SINGLETON,
            config=config.get("training", {}),
        )
        self.container.register(
            ExchangeFactory,
            ExchangeFactory,
            lifetime=ServiceLifetime.SINGLETON,
            config=config.get("exchange", {}),
        )

    def get_registered_services(self) -> dict[str, Any]:
        """Return the currently registered services."""

        return self.container.get_all_services()

    def validate_registrations(self) -> bool:
        """Ensure that the mandatory services are present."""

        required_services = [IEventBus, IAnalyst, IStrategist, ITactician, ISupervisor]
        missing_services: list[str] = []

        registered = self.container.get_all_services()
        for service in required_services:
            if service not in registered:
                missing_services.append(service.__name__)

        if missing_services:
            self.logger.error("Missing required services: %s", missing_services)
            return False

        self.logger.info("All required services are registered")
        return True

