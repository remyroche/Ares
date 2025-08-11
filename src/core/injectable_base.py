# src/core/injectable_base.py

"""
Base classes for dependency injection support.

This module provides base classes that make it easy for trading components
to participate in the dependency injection system.
"""

from abc import ABC
from typing import Any

from src.core.dependency_injection import Configurable, Injectable
from src.interfaces.base_interfaces import (
    IEventBus,
    IExchangeClient,
    IPerformanceReporter,
    IStateManager,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    warning,
)


class InjectableBase(Injectable, Configurable, ABC):
    """
    Base class for all injectable trading components.

    Provides common dependency injection functionality and configuration support.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.logger = system_logger.getChild(self.__class__.__name__)
        self._initialized = False

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the component with provided configuration."""
        self.config.update(config)
        self.logger.debug(f"Component {self.__class__.__name__} configured")

    async def initialize(self) -> bool:
        """Initialize the component. Override in subclasses for custom initialization."""
        if self._initialized:
            return True

        self.logger.info(f"Initializing {self.__class__.__name__}")
        self._initialized = True
        return True

    async def shutdown(self) -> None:
        """Shutdown the component. Override in subclasses for custom cleanup."""
        self.logger.info(f"Shutting down {self.__class__.__name__}")
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        return self._initialized


class TradingComponentBase(InjectableBase):
    """
    Base class for core trading components (Analyst, Strategist, Tactician, Supervisor).

    Provides common dependencies and functionality needed by all trading components.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        exchange_client: IExchangeClient | None = None,
        state_manager: IStateManager | None = None,
        event_bus: IEventBus | None = None,
    ):
        super().__init__(config)

        # Core dependencies (will be injected)
        self.exchange_client = exchange_client
        self.state_manager = state_manager
        self.event_bus = event_bus

        # Component state
        self.is_running = False

    async def start(self) -> None:
        """Start the trading component."""
        if not self._initialized:
            await self.initialize()

        if self.is_running:
            self.print(warning("{self.__class__.__name__} is already running"))
            return

        self.logger.info(f"Starting {self.__class__.__name__}")
        self.is_running = True

        # Perform component-specific startup
        await self._start_component()

    async def stop(self) -> None:
        """Stop the trading component."""
        if not self.is_running:
            self.print(warning("{self.__class__.__name__} is not running"))
            return

        self.logger.info(f"Stopping {self.__class__.__name__}")
        self.is_running = False

        # Perform component-specific shutdown
        await self._stop_component()

    async def _start_component(self) -> None:
        """Override in subclasses for component-specific startup logic."""

    async def _stop_component(self) -> None:
        """Override in subclasses for component-specific shutdown logic."""

    def _validate_dependencies(self) -> bool:
        """Validate that all required dependencies are available."""
        if not self.exchange_client:
            self.print(error("Exchange client dependency not available"))
            return False

        if not self.state_manager:
            self.print(error("State manager dependency not available"))
            return False

        return True


class AnalystBase(TradingComponentBase):
    """Base class for analyst implementations with dependency injection support."""

    async def initialize(self) -> bool:
        """Initialize analyst with dependency validation."""
        if not await super().initialize():
            return False

        if not self._validate_dependencies():
            return False

        self.logger.info("Analyst initialized successfully")
        return True


class StrategistBase(TradingComponentBase):
    """Base class for strategist implementations with dependency injection support."""

    async def initialize(self) -> bool:
        """Initialize strategist with dependency validation."""
        if not await super().initialize():
            return False

        if not self._validate_dependencies():
            return False

        self.logger.info("Strategist initialized successfully")
        return True


class TacticianBase(TradingComponentBase):
    """Base class for tactician implementations with dependency injection support."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        exchange_client: IExchangeClient | None = None,
        state_manager: IStateManager | None = None,
        event_bus: IEventBus | None = None,
        performance_reporter: IPerformanceReporter | None = None,
    ):
        super().__init__(config, exchange_client, state_manager, event_bus)
        self.performance_reporter = performance_reporter

    async def initialize(self) -> bool:
        """Initialize tactician with dependency validation."""
        if not await super().initialize():
            return False

        if not self._validate_dependencies():
            return False

        if not self.performance_reporter:
            self.print(error("Performance reporter dependency not available"))
            return False

        self.logger.info("Tactician initialized successfully")
        return True


class SupervisorBase(TradingComponentBase):
    """Base class for supervisor implementations with dependency injection support."""

    async def initialize(self) -> bool:
        """Initialize supervisor with dependency validation."""
        if not await super().initialize():
            return False

        if not self._validate_dependencies():
            return False

        if not self.event_bus:
            self.print(error("Event bus dependency not available"))
            return False

        self.logger.info("Supervisor initialized successfully")
        return True
