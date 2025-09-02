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
        """Initialize the signal handler with configuration."""
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("SignalHandler")

        # Signal management
        self.registered_handlers: dict[int, Callable] = {}
        self.shutdown_callbacks: list[Callable] = []
        self.is_shutting_down: bool = False

        # Configuration
        self.signal_config: dict[str, Any] = self.config.get("signal_handler", {})
        self.graceful_shutdown_timeout: int = self.signal_config.get(
            "graceful_shutdown_timeout", 30
        )
        self.enable_signal_handling: bool = self.signal_config.get(
            "enable_signal_handling", True
        )

    def print(self, message: Any) -> None:
        """Print message with proper logging."""
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
    async def initialize(self) -> bool:
        """Initialize the signal handler."""
        try:
            self.logger.info("Initializing Signal Handler...")

            # Load signal configuration
            await self._load_signal_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.print(invalid("Invalid configuration for signal handler"))
                return False

            # Register signal handlers
            if self.enable_signal_handling:
                await self._register_signal_handlers()

            self.logger.info("✅ Signal Handler initialization completed successfully")
            return True

        except Exception as e:
            self.print(failed(f"❌ Signal Handler initialization failed: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="signal configuration loading",
    )
    async def _load_signal_configuration(self) -> None:
        """Load and validate signal configuration."""
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

        except Exception as e:
            self.print(error(f"Error loading signal configuration: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate the signal handler configuration."""
        try:
            # Validate shutdown timeout
            if self.graceful_shutdown_timeout <= 0:
                self.print(invalid("Invalid graceful shutdown timeout"))
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.print(error(f"Error validating configuration: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="signal handler registration",
    )
    async def _register_signal_handlers(self) -> None:
        """Register signal handlers for various signals."""
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

        except Exception as e:
            self.print(error(f"Error registering signal handlers: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="signal handler registration",
    )
    def _register_handler(self, sig: int, handler: Callable) -> None:
        """Register a signal handler."""
        try:
            # Store original handler if exists
            original_handler = signal.getsignal(sig)
            self.registered_handlers[sig] = original_handler

            # Register new handler
            signal.signal(sig, handler)

        except Exception as e:
            self.print(error(f"Error registering signal handler for {sig}: {e}"))

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid signal handling"),
            AttributeError: (None, "Missing signal components"),
            KeyError: (None, "Missing required signal data"),
        },
        default_return=None,
        context="SIGTERM handling",
    )
    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        """Handle SIGTERM signal."""
        try:
            self.print(warning("🛑 Received SIGTERM signal"))
            self._initiate_shutdown("SIGTERM")

        except Exception as e:
            self.print(error(f"Error handling SIGTERM: {e}"))

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid signal handling"),
            AttributeError: (None, "Missing signal components"),
            KeyError: (None, "Missing required signal data"),
        },
        default_return=None,
        context="SIGINT handling",
    )
    def _handle_sigint(self, signum: int, frame: Any) -> None:
        """Handle SIGINT signal."""
        try:
            self.print(warning("🛑 Received SIGINT signal"))
            self._initiate_shutdown("SIGINT")

        except Exception as e:
            self.print(error(f"Error handling SIGINT: {e}"))

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid signal handling"),
            AttributeError: (None, "Missing signal components"),
            KeyError: (None, "Missing required signal data"),
        },
        default_return=None,
        context="SIGHUP handling",
    )
    def _handle_sighup(self, signum: int, frame: Any) -> None:
        """Handle SIGHUP signal for configuration reload."""
        try:
            self.logger.info("🔄 Received SIGHUP signal - reloading configuration")

            # Import config module
            from src.config import CONFIG, load_configuration

            # Reload configuration from file
            self.logger.info("📋 Reloading configuration from config file...")
            new_config = load_configuration()

            if new_config:
                # Update global CONFIG
                CONFIG.clear()
                CONFIG.update(new_config)
                self.logger.info("✅ Configuration reloaded successfully")

                # Notify components about configuration change
                self._notify_configuration_change()
            else:
                self.print(failed("❌ Failed to reload configuration"))

        except Exception as e:
            self.print(error(f"Error handling SIGHUP: {e}"))

    def _notify_configuration_change(self) -> None:
        """Notify components about configuration change."""
        try:
            self.logger.info("📢 Notifying components about configuration change...")

            # This would typically involve calling callbacks or updating component states
            # For now, we'll just log the notification
            self.logger.info("✅ Configuration change notification sent")

        except Exception as e:
            self.print(error(f"Error notifying configuration change: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="shutdown initiation",
    )
    def _initiate_shutdown(self, reason: str) -> None:
        """Initiate graceful shutdown."""
        try:
            if self.is_shutting_down:
                self.logger.info("Shutdown already in progress")
                return

            self.is_shutting_down = True
            self.print(
                initialization_error(f"🛑 Initiating graceful shutdown: {reason}")
            )

            # Run shutdown callbacks
            self._run_shutdown_callbacks()

        except Exception as e:
            self.print(initialization_error(f"Error initiating shutdown: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="shutdown callbacks execution",
    )
    def _run_shutdown_callbacks(self) -> None:
        """Execute all registered shutdown callbacks."""
        try:
            if not self.shutdown_callbacks:
                self.logger.info("No shutdown callbacks registered")
                return

            self.logger.info(
                f"Running {len(self.shutdown_callbacks)} shutdown callbacks..."
            )

            for i, callback in enumerate(self.shutdown_callbacks):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.run(callback())  # Changed to asyncio.run to handle coroutines
                    else:
                        callback()
                    self.logger.info(f"✅ Shutdown callback {i + 1} completed")
                except Exception as e:
                    self.print(failed(f"❌ Shutdown callback {i + 1} failed: {e}"))

            self.logger.info("✅ All shutdown callbacks completed")

        except Exception as e:
            self.print(error(f"Error running shutdown callbacks: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="shutdown callback registration",
    )
    def register_shutdown_callback(self, callback: Callable) -> None:
        """Register a shutdown callback."""
        try:
            if callback not in self.shutdown_callbacks:
                self.shutdown_callbacks.append(callback)
                self.logger.info("Shutdown callback registered")
            else:
                self.print(warning("Shutdown callback already registered"))

        except Exception as e:
            self.print(error(f"Error registering shutdown callback: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="shutdown callback removal",
    )
    def unregister_shutdown_callback(self, callback: Callable) -> None:
        """Unregister a shutdown callback."""
        try:
            if callback in self.shutdown_callbacks:
                self.shutdown_callbacks.remove(callback)
                self.logger.info("Shutdown callback unregistered")
            else:
                self.print(missing("Shutdown callback not found"))

        except Exception as e:
            self.print(error(f"Error unregistering shutdown callback: {e}"))

    def get_signal_status(self) -> dict[str, Any]:
        """Get current signal handler status."""
        return {
            "is_shutting_down": self.is_shutting_down,
            "enable_signal_handling": self.enable_signal_handling,
            "graceful_shutdown_timeout": self.graceful_shutdown_timeout,
            "registered_handlers": list(self.registered_handlers.keys()),
            "shutdown_callbacks_count": len(self.shutdown_callbacks),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="signal handler cleanup",
    )
    async def stop(self) -> None:
        """Stop the signal handler and restore original handlers."""
        self.logger.info("🛑 Stopping Signal Handler...")

        try:
            # Restore original signal handlers
            for sig, original_handler in self.registered_handlers.items():
                try:
                    signal.signal(sig, original_handler)
                    self.logger.info(f"Restored original handler for signal {sig}")
                except Exception as e:
                    self.logger.warning(
                        f"Could not restore handler for signal {sig}: {e}"
                    )

            self.logger.info("✅ Signal Handler stopped successfully")

        except Exception as e:
            self.print(error(f"Error stopping signal handler: {e}"))


# Global signal handler instance
signal_handler: SignalHandler | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="signal handler setup",
)
async def setup_signal_handler(config: dict[str, Any] | None = None) -> SignalHandler | None:
    """Set up the global signal handler."""
    try:
        if config is None:
            # Fallback implementation for config
            config = {}

        signal_handler = SignalHandler(config)
        success = await signal_handler.initialize()

        if success:
            print("✅ Signal handler setup completed successfully")
            return signal_handler
        print(failed("Signal handler setup failed"))
        return None

    except Exception as e:
        print(failed(f"Signal handler setup failed: {e}"))
        return None


class GracefulShutdown:
    """
    Context manager for graceful shutdown handling.
    """

    def __init__(self):
        self.signal_handler = signal_handler
        self.original_handlers = {}

    def __enter__(self):
        """Set up graceful shutdown handlers."""
        if self.signal_handler:
            # Store original handlers
            self.original_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
            self.original_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)

            # Set up new handlers
            signal.signal(signal.SIGTERM, self.signal_handler._handle_sigterm)
            signal.signal(signal.SIGINT, self.signal_handler._handle_sigint)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original signal handlers."""
        if self.signal_handler:
            # Restore original handlers
            for sig, handler in self.original_handlers.items():
                signal.signal(sig, handler)


def setup_signal_handlers() -> SignalHandler:
    """Set up signal handlers for backward compatibility."""
    config = {
        "signal_handler": {
            "enable_signal_handling": True,
            "graceful_shutdown_timeout": 30,
            "handle_sigterm": True,
            "handle_sigint": True,
            "handle_sighup": False,
        },
    }

    # Create and initialize signal handler
    signal_handler = SignalHandler(config)

    # Initialize synchronously for backward compatibility
    import asyncio

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(signal_handler.initialize())
    except Exception as e:
        print(f"Warning: Signal handler initialization failed: {e}")

    return signal_handler
