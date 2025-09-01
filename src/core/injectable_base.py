# src/core/injectable_base.py

"""
Base classes for dependency injection support.

This module provides base classes that make it easy for trading components
to participate in the dependency injection system.
"""


import sys
from typing import TYPE_CHECKING, Any

from src.utils.logger import system_logger

if TYPE_CHECKING:  # only for type checking; avoids runtime import cost
from src.interfaces.base_interfaces import (
IEventBus,
IExchangeClient,
IStateManager,
)


class InjectableBase:
    """
Base class for all injectable trading components.

Provides common dependency injection functionality and configuration support.
"""

def __init__(self, config: dict[str, Any] | None = None):
        self.config: dict[str, Any] = config or {}
self.logger = system_logger.getChild(self.__class__.__name__)
self._initialized: bool = False
# Provide a safe print shim so subclasses can call self.print
if not hasattr(self, "print"):
            def _shim_print(message: str) -> None:
                try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
self.logger.error(str(message))
except Exception as e:  # noqa: BLE001 - fallback safety
print(
f"Logger failed in shim_print: {e}",
file=sys.stderr,
)
print(f"Original message: {message}", file=sys.stderr)

self.print = _shim_print  # type: ignore[attr-defined]

def configure(self, config: dict[str, Any]) -> None:
        """Configure the component with provided configuration."""
self.config.update(config)
self.logger.debug(
"Component %s configured",
self.__class__.__name__,
)

async def initialize(self) -> bool:
        """Initialize the component.

Override in subclasses for custom initialization.
"""
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
Base class for core trading components (Analyst, Strategist,
Tactician, Supervisor).

Provides common dependencies and functionality needed by all trading
components.
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
self.exchange_client: IExchangeClient | None = exchange_client
self.state_manager: IStateManager | None = state_manager
self.event_bus: IEventBus | None = event_bus

# Component state
self.is_running: bool = False

async def start(self) -> None:
        """Start the trading component."""
if self.is_running:
            return

self.logger.info(f"Starting {self.__class__.__name__}")
self.is_running = True

async def stop(self) -> None:
        """Stop the trading component."""
if not self.is_running:
            return

self.logger.info(f"Stopping {self.__class__.__name__}")
self.is_running = False

@property
def is_active(self) -> bool:
        """Check if component is active and running."""
return self.is_running and self.is_initialized
