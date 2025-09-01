# src/core/service_registry.py

"""
Service registry for dependency injection container configuration.

This module provides centralized service registration for all trading components, ensuring proper dependency injection throughout the system.
"""

from src.core.dependency_injection import DependencyContainer, ServiceLifetime
from src.utils.logger import system_logger
from typing import Any
from exchange.factory import ExchangeFactory
from src.training.training_manager import TrainingManager
from src.analyst.analyst import Analyst
from src.components.modular_analyst import ModularAnalyst
from src.components.modular_strategist import ModularStrategist
from src.components.modular_tactician import ModularTactician
from src.interfaces.event_bus import EventBus
from src.strategist.strategist import Strategist
from src.supervisor.supervisor import Supervisor
from src.tactician.tactician import Tactician
from src.interfaces.base_interfaces import (
    IAnalyst,
    IEventBus,
    IStrategist,
    ISupervisor,
    ITactician,
)


