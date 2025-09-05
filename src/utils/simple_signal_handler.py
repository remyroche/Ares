from typing import Dict, List, Optional, Union, Any, Tuple
"""
Simple Signal Handler for Ares Trading System

This module provides basic signal handling functionality without complex decorators.
"""
import logging
import signal
import sys
from typing import Callable

class SimpleSignalHandler:
    """Simple signal handler for graceful shutdown."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.shutdown_callbacks: list[Callable] = []
        self.initialized = False

    def register_shutdown_callback(self, callback: Callable) -> None:
        """Register a callback to be called on shutdown."""
        self.shutdown_callbacks.append(callback)

    def initialize(self) -> bool:
        """Initialize signal handlers."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self.initialized = True
            self.logger.info('Signal handlers initialized successfully')
            return True
        except Exception as e:
            self.logger.error(f'Failed to initialize signal handlers: {e}')
            return False

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        self.logger.info(f'Received signal {signal_name}, initiating graceful shutdown...')
        for callback in self.shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f'Error in shutdown callback: {e}')
        sys.exit(0)

def setup_signal_handlers() -> SimpleSignalHandler:
    """Setup signal handlers."""
    handler = SimpleSignalHandler()
    handler.initialize()
    return handler