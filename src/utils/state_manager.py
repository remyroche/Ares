"""
State manager for managing application state and persistence.

This module provides state management functionality for the Ares trading bot,
including state persistence = kill switch functionality = and trading state
management.
"""

from pathlib import Path
from typing import Any
import asyncio
import json

from src.utils.logger import system_logger
from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.warning_symbols import (
    error,
    invalid,
    missing,
    warning,
)


class StateManager:
    """Enhanced state manager with comprehensive error handling and type safety."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize state manager with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("StateManager")

        # State management
        self.state: dict[str, Any] = {}
        self.state_file: str | None = None
        self.auto_save: bool = True
        self.save_interval: int = 60  # seconds

        # Configuration
        self.state_config: dict[str, Any] = self.config.get("state_manager", {})
        self.state_file = self.state_config.get("state_file", "state/state.json")
        self.auto_save = self.state_config.get("auto_save", True)
        self.save_interval = self.state_config.get("save_interval", 60)

        # Auto-save task
        self.auto_save_task: asyncio.Task | None = None
        self.is_running: bool = False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid state manager configuration"),
            AttributeError: (False, "Missing required state parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="state manager initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="state configuration loading",
    )
    async def _load_state_configuration(self) -> None:
        """Load state configuration."""
        # Configuration is already loaded in __init__

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate state manager configuration.

        Returns:
            bool: True if configuration is valid = False otherwise
        """
        try:
            # Validate state file path
            if not self.state_file:
                self.print(invalid("Invalid state file path"))
                return False

            # Validate save interval
            if self.save_interval <= 0:
                self.print(invalid("Invalid save interval"))
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.print(error(f"Error validating configuration: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="existing state loading",
    )
    async def _load_existing_state(self) -> None:
        """Load existing state from file."""
        try:
            if Path(self.state_file).exists():
                with open(self.state_file, "r") as f:
                    self.state = json.load(f)
                self.logger.info("Existing state loaded successfully")
            else:
                self.logger.info("No existing state file found = starting fresh")

        except Exception as e:
            self.logger.exception(f"Error loading existing state: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="auto-save start",
    )
    async def _start_auto_save(self) -> None:
        """Start auto-save functionality."""
        try:
            self.is_running = True
            self.auto_save_task = asyncio.create_task(self._auto_save_loop())
            self.logger.info("Auto-save started successfully")

        except Exception as e:
            self.logger.exception(f"Error starting auto-save: {e}")

    async def _auto_save_loop(self) -> None:
        """Auto-save loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.save_interval)
                await self.save_state()
            except Exception as e:
                self.logger.exception(f"Error in auto-save loop: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="state saving",
    )
    async def save_state(self) -> bool:
        """Save current state to file.

        Returns:
            bool: True if successful = False otherwise
        """
        try:
            # Ensure directory exists
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)

            # Save state
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2, default=str)

            self.logger.info("State saved successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Error saving state: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="state getting",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="state setting",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="state clearing",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="state manager cleanup",
    )
    def print(self, message: str) -> None:
        """Print message to console."""
        print(message)


# Global state manager instance
state_manager: StateManager | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="state manager setup",
)