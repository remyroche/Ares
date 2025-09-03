from __future__ import annotations
"""
State manager for managing application state and persistence.

This module provides state management functionality for the Ares trading bot,
including state persistence = kill switch functionality = and trading state
management.
"""

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    invalid,
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

    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid state manager configuration"),
            AttributeError: (False, "Missing required state parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="state manager initialization",
    )
    async def initialize(self) -> bool:
        """Initialize state manager with enhanced error handling.

        Returns:
            bool: True if initialization successful = False otherwise
        """
        self.logger.info("Initializing State Manager...")

        # Load state configuration
        await self._load_state_configuration()

        # Validate configuration
        if not self._validate_configuration():
            self.print(invalid("Invalid configuration for state manager"))
            return False

        # Load existing state
        await self._load_existing_state()

        # Start auto-save if enabled
        if self.auto_save:
            await self._start_auto_save()

        self.logger.info("✅ State Manager initialization completed successfully")
        return True

    @handles_errors(fallback=None)
    async def _load_state_configuration(self) -> None:
        """Load state configuration."""
        # Configuration is already loaded in __init__

    @handles_errors(fallback=False)
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

    @handles_errors(fallback=None)
    async def _load_existing_state(self) -> None:
        """Load existing state from file."""
        try:
            if Path(self.state_file).exists():
                with open(self.state_file) as f:
                    self.state = json.load(f)
                self.logger.info("Existing state loaded successfully")
            else:
                self.logger.info("No existing state file found = starting fresh")

        except Exception as e:
            self.logger.exception(f"Error loading existing state: {e}")

    @handles_errors(fallback=None)
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

    @handles_errors(fallback=False)
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

    @handles_errors(fallback=None)
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            Any: State value
        """
        try:
            return self.state.get(key, default)
        except Exception as e:
            self.logger.exception(f"Error getting state: {e}")
            return default

    @handles_errors(fallback=None)
    def set_state(self, key: str, value: Any) -> None:
        """Set state value.

        Args:
            key: State key
            value: State value
        """
        try:
            self.state[key] = value
            self.logger.debug(f"State updated: {key} = {value}")
        except Exception as e:
            self.logger.exception(f"Error setting state: {e}")

    @handles_errors(fallback=None)
    def clear_state(self) -> None:
        """Clear all state."""
        try:
            self.state.clear()
            self.logger.info("State cleared successfully")
        except Exception as e:
            self.logger.exception(f"Error clearing state: {e}")

    @handles_errors(fallback=None)
    async def stop(self) -> None:
        """Stop the state manager."""
        self.logger.info("🛑 Stopping State Manager...")

        try:
            # Stop auto-save
            self.is_running = False
            if self.auto_save_task:
                self.auto_save_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.auto_save_task

            # Save final state
            await self.save_state()

            self.logger.info("✅ State Manager stopped successfully")

        except Exception as e:
            self.logger.exception(f"Error stopping state manager: {e}")

    def print(self, message: str) -> None:
        """Print message to console."""
        print(message)

# Global state manager instance
state_manager: StateManager | None = None

@handles_errors(fallback=None)
async def setup_state_manager(
    config: dict[str, Any] | None = None,
) -> StateManager | None:
    """Setup global state manager.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[StateManager]: Global state manager instance
    """
    try:
        global state_manager

        if config is None:
            # Fallback implementation for config
            config = {
                "state_manager": {
                    "state_file": "state/state.json",
                    "auto_save": True,
                    "save_interval": 60,
                },
            }

        # Create state manager
        state_manager = StateManager(config)

        # Initialize state manager
        success = await state_manager.initialize()
        if success:
            return state_manager
        return None

    except Exception:
        return None
