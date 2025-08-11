"""
State manager for managing application state and persistence.

This module provides state management functionality for the Ares trading bot,
including state persistence, kill switch functionality, and trading state
management.
"""

import asyncio
import contextlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    invalid,
    missing,
    warning,
)


class StateManager:
    """
    Enhanced state manager with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize state manager with enhanced type safety.

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
    async def initialize(self) -> bool:
        """
        Initialize state manager with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
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
        """Validate state manager configuration."""
        if not self.state_file:
            self.print(error("State file not configured"))
            return False

        if self.save_interval <= 0:
            self.print(invalid("Invalid save interval"))
            return False

        return True

    @handle_file_operations(
        default_return=None,
        context="existing state loading",
    )
    async def _load_existing_state(self) -> None:
        """Load existing state from file."""
        if not self.state_file:
            return

        state_path = Path(self.state_file)
        if state_path.exists():
            try:
                with state_path.open("r") as f:
                    self.state = json.load(f)
                self.logger.info(f"State loaded from: {self.state_file}")
            except Exception:
                self.print(warning("Could not load existing state: {e}"))
                self.state = {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="auto-save task start",
    )
    async def _start_auto_save(self) -> None:
        """Start auto-save task."""
        self.is_running = True
        self.auto_save_task = asyncio.create_task(self._auto_save_loop())
        self.logger.info("Auto-save task started")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="auto-save loop",
    )
    async def _auto_save_loop(self) -> None:
        """Auto-save loop."""
        while self.is_running:
            await asyncio.sleep(self.save_interval)
            if self.is_running:
                await self.save_state()

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid state key"),
            AttributeError: (False, "Missing state component"),
            KeyError: (False, "Missing required state data"),
        },
        default_return=None,
        context="state setting",
    )
    async def set_state(self, key: str, value: Any) -> bool:
        """
        Set state value.

        Args:
            key: State key
            value: State value

        Returns:
            bool: True if successful, False otherwise
        """
        if not key:
            self.print(invalid("Invalid state key"))
            return False

        # Update state
        self.state[key] = value

        # Add to history
        self.state.setdefault("history", []).append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "action": "set",
                "key": key,
                "value": value,
            },
        )

        # Auto-save if enabled
        if self.auto_save:
            await self.save_state()

        self.logger.info(f"State updated: {key}")
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="state getting",
    )
    async def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get state value.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            Any: State value or default
        """
        if not key:
            self.print(invalid("Invalid state key"))
            return default

        return self.state.get(key, default)

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="state deletion",
    )
    async def delete_state(self, key: str) -> bool:
        """
        Delete state value.

        Args:
            key: State key

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not key:
                self.print(invalid("Invalid state key"))
                return False

            if key in self.state:
                del self.state[key]

                # Add to history
                self.state.setdefault("history", []).append(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "action": "delete",
                        "key": key,
                    },
                )

                # Auto-save if enabled
                if self.auto_save:
                    await self.save_state()

                self.logger.info(f"State deleted: {key}")
                return True
            self.print(missing("State key not found: {key}"))
            return False

        except Exception:
            self.print(error("Error deleting state: {e}"))
            return False

    @handle_file_operations(
        default_return=None,
        context="state saving",
    )
    async def save_state(self) -> None:
        """Save state to file."""
        try:
            # Ensure directory exists
            state_path = Path(self.state_file)
            state_dir = state_path.parent
            if state_dir and not state_dir.exists():
                state_dir.mkdir(parents=True, exist_ok=True)

            # Save state
            with state_path.open("w") as f:
                json.dump(self.state, f, indent=2, default=str)

            self.logger.info(f"State saved to: {self.state_file}")

        except Exception:
            self.print(error("Error saving state: {e}"))

    @handle_file_operations(
        default_return=None,
        context="state backup creation",
    )
    async def create_backup(self) -> None:
        """Create backup of state file."""
        try:
            state_path = Path(self.state_file)
            if not state_path.exists():
                self.print(warning("No state file to backup"))
                return

            # Create backup filename
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            backup_file = f"{self.state_file}.backup_{timestamp}"

            # Copy state file
            shutil.copy2(self.state_file, backup_file)

            self.logger.info(f"State backup created: {backup_file}")

        except Exception:
            self.print(error("Error creating state backup: {e}"))

    def get_state_status(self) -> dict[str, Any]:
        """
        Get state manager status information.

        Returns:
            Dict[str, Any]: State manager status
        """
        return {
            "is_running": self.is_running,
            "auto_save": self.auto_save,
            "save_interval": self.save_interval,
            "state_file": self.state_file,
            "state_keys": list(self.state.keys()),
            "state_size": len(json.dumps(self.state)),
            "history_count": len(self.state.get("history", [])),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="state manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the state manager."""
        self.logger.info("🛑 Stopping State Manager...")

        try:
            # Stop auto-save task
            self.is_running = False
            if self.auto_save_task:
                self.auto_save_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.auto_save_task

            # Save final state
            await self.save_state()

            self.logger.info("✅ State Manager stopped successfully")

        except Exception:
            self.print(error("Error stopping state manager: {e}"))


# Global state manager instance
state_manager: StateManager | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="state manager setup",
)
async def setup_state_manager(
    config: dict[str, Any] | None = None,
) -> StateManager | None:
    """
    Setup global state manager.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[StateManager]: Global state manager instance
    """
    try:
        global state_manager

        if config is None:
            config = {
                "state_manager": {
                    "state_file": "state/state.json",
                    "auto_save": True,
                    "save_interval": 60,
                    "backup_enabled": True,
                    "max_backups": 5,
                },
            }

        # Create state manager
        state_manager = StateManager(config)

        # Initialize state manager
        success = await state_manager.initialize()
        if success:
            return state_manager
        return None

    except Exception as e:
        print(f"Error setting up state manager: {e}")
        return None
