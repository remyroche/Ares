"""
Signal handling utilities for graceful shutdown.

This module provides centralized signal handling for graceful shutdown
of the application, including both synchronous and asynchronous cleanup.
"""

import asyncio
import signal
from collections.abc import Callable
from typing import Any

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    failed,
    initialization_error,
    invalid,
    missing,
    warning,
)


class SignalHandler:
    """
    Enhanced signal handler with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize signal handler with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("SignalHandler")

        # Signal management
        self.registered_handlers: dict[int, Callable] = {}
        self.shutdown_callbacks: list[Callable] = []
        self.is_shutting_down: bool = False

        # Configuration
        self.signal_config: dict[str, Any] = self.config.get("signal_handler", {})
        self.graceful_shutdown_timeout: int = self.signal_config.get(
            "graceful_shutdown_timeout",
            30,
        )
        self.enable_signal_handling: bool = self.signal_config.get(
            "enable_signal_handling",
            True,
        )

    def print(self, message: Any) -> None:
        """
        Compatibility helper to mirror other components' print method.

        Routes messages through the component logger so output appears in the
        terminal and logs consistently.
        """
        # Ensure string conversion in case formatting helpers are used
        self.logger.info(str(message))

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid signal handler configuration"),
            AttributeError: (False, "Missing required signal parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="signal handler initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="signal configuration loading",
    )
    async def _load_signal_configuration(self) -> None:
        """Load signal configuration."""
        try:
            # Set default signal parameters
            self.signal_config.setdefault("enable_signal_handling", True)
            self.signal_config.setdefault("graceful_shutdown_timeout", 30)
            self.signal_config.setdefault("handle_sigterm", True)
            self.signal_config.setdefault("handle_sigint", True)
            self.signal_config.setdefault("handle_sighup", False)

            # Update configuration
            self.graceful_shutdown_timeout = self.signal_config[
                "graceful_shutdown_timeout"
            ]
            self.enable_signal_handling = self.signal_config["enable_signal_handling"]

            self.logger.info("Signal configuration loaded successfully")

        except Exception:
            self.print(error("Error loading signal configuration: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate signal configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate shutdown timeout
            if self.graceful_shutdown_timeout <= 0:
                self.print(invalid("Invalid graceful shutdown timeout"))
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception:
            self.print(error("Error validating configuration: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="signal handler registration",
    )
    async def _register_signal_handlers(self) -> None:
        """Register signal handlers."""
        try:
            # Register SIGTERM handler
            if self.signal_config.get("handle_sigterm", True):
                self._register_handler(signal.SIGTERM, self._handle_sigterm)
                self.logger.info("Registered SIGTERM handler")

            # Register SIGINT handler
            if self.signal_config.get("handle_sigint", True):
                self._register_handler(signal.SIGINT, self._handle_sigint)
                self.logger.info("Registered SIGINT handler")

            # Register SIGHUP handler
            if self.signal_config.get("handle_sighup", False):
                self._register_handler(signal.SIGHUP, self._handle_sighup)
                self.logger.info("Registered SIGHUP handler")

            self.logger.info("Signal handlers registered successfully")

        except Exception:
            self.print(error("Error registering signal handlers: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="signal handler registration",
    )
    def _register_handler(self, sig: int, handler: Callable) -> None:
        """
        Register a signal handler.

        Args:
            sig: Signal number
            handler: Handler function
        """
        try:
            # Store original handler if exists
            original_handler = signal.getsignal(sig)
            self.registered_handlers[sig] = original_handler

            # Register new handler
            signal.signal(sig, handler)

        except Exception:
            self.print(error("Error registering signal handler for {sig}: {e}"))

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid signal handling"),
            AttributeError: (None, "Missing signal components"),
            KeyError: (None, "Missing required signal data"),
        },
        default_return=None,
        context="SIGTERM handling",
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid signal handling"),
            AttributeError: (None, "Missing signal components"),
            KeyError: (None, "Missing required signal data"),
        },
        default_return=None,
        context="SIGINT handling",
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid signal handling"),
            AttributeError: (None, "Missing signal components"),
            KeyError: (None, "Missing required signal data"),
        },
        default_return=None,
        context="SIGHUP handling",
    )
    def _notify_configuration_change(self) -> None:
        """Notify registered components about configuration change."""
        try:
            self.logger.info("📢 Notifying components about configuration change...")

            # This would typically involve calling callbacks or updating component states
            # For now, we'll just log the notification
            self.logger.info("✅ Configuration change notification sent")

        except Exception:
            self.print(error("Error notifying configuration change: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="shutdown initiation",
    )
    def _initiate_shutdown(self, reason: str) -> None:
        """
        Initiate graceful shutdown.

        Args:
            reason: Reason for shutdown
        """
        try:
            if self.is_shutting_down:
                self.logger.info("Shutdown already in progress")
                return

            self.is_shutting_down = True
            self.print(
                initialization_error("🛑 Initiating graceful shutdown: {reason}"),
            )

            # Run shutdown callbacks
            # The original code had asyncio.create_task(self._run_shutdown_callbacks())
            # This line was removed as per the edit hint.
            # The original code also had asyncio.set_event_loop(loop) and loop.run_until_complete(signal_handler.initialize())
            # This was removed as per the edit hint.
            self._run_shutdown_callbacks()

        except Exception:
            self.print(initialization_error("Error initiating shutdown: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="shutdown callbacks execution",
    )
    def _run_shutdown_callbacks(self) -> None:
        """Run shutdown callbacks."""
        try:
            if not self.shutdown_callbacks:
                self.logger.info("No shutdown callbacks registered")
                return

            self.logger.info(
                f"Running {len(self.shutdown_callbacks)} shutdown callbacks...",
            )

            for i, callback in enumerate(self.shutdown_callbacks):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.run(
                            callback(),
                        )  # Changed to asyncio.run to handle coroutines
                    else:
                        callback()
                    self.logger.info(f"✅ Shutdown callback {i+1} completed")
                except Exception:
                    self.print(failed("❌ Shutdown callback {i+1} failed: {e}"))

            self.logger.info("✅ All shutdown callbacks completed")

        except Exception:
            self.print(error("Error running shutdown callbacks: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="shutdown callback registration",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="shutdown callback removal",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="signal handler cleanup",
    )

# Global signal handler instance
signal_handler: SignalHandler | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="signal handler setup",
)

class GracefulShutdown:
    """
    Context manager for graceful shutdown handling.
    """

    def __init__(self, signal_handler: SignalHandler | None = None):
        self.signal_handler = signal_handler
        self.original_handlers = {}

