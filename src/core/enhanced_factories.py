# src/core/enhanced_factories.py

"""
Enhanced factory classes that use dependency injection.

This module provides factory classes that create trading components
using proper dependency injection patterns.
"""

from src.database.firestore_manager import FirestoreManager
from src.supervisor.performance_reporter import PerformanceReporter
from src.core.dependency_injection import DependencyContainer
from src.utils.logger import system_logger
from typing import Any
from src.database.influxdb_manager import InfluxDBManager
from exchange.factory import ExchangeFactory
from src.utils.state_manager import StateManager
from src.interfaces.base_interfaces import (
    IAnalyst,
    IExchangeClient,
    IPerformanceReporter,
    IStateManager,
    IStrategist,
    ISupervisor,
    ITactician,
)
from src.utils.warning_symbols import failed


class TradingSystemFactory:
    """
    Factory for creating complete trading systems with dependency injection.
    """

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("TradingSystemFactory")


class ExchangeClientFactory:
    """
    Factory for creating exchange clients with dependency injection support.
    """

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("ExchangeClientFactory")


class DatabaseFactory:
    """
    Factory for creating database managers with dependency injection support.
    """

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("DatabaseFactory")


class StateManagerFactory:
    """
    Factory for creating state managers with dependency injection support.
    """

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("StateManagerFactory")


class PerformanceReporterFactory:
    """
    Factory for creating performance reporters with dependency injection support.
    """

    def __init__(self, container: DependencyContainer):
        self.container = container
        self.logger = system_logger.getChild("PerformanceReporterFactory")
