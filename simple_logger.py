#!/usr/bin/env python3

"""
Simple logger replacement to bypass initialization issues
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def _get_tprint():
    """Get tprint function with lazy import to avoid circular dependencies."""
    try:
        from src.utils.tprint import tprint
        return tprint
    except ImportError:
        # Fallback to regular print if tprint is not available
        def fallback_print(*args, **kwargs):
            if args:
                print(f"[SIMPLE_LOGGER] {args[0]}", *args[1:], **kwargs)
            else:
                print("[SIMPLE_LOGGER]", **kwargs)
        return fallback_print

# Create a simple logger that works
def create_simple_logger():
    """Create a basic logger without complex initialization."""
    tprint = _get_tprint()
    logger = logging.getLogger('AresSimple')
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Create the logger instance
tprint = _get_tprint()
tprint("🔧 [SIMPLE_LOGGER] Creating simple logger instance...")
system_logger = create_simple_logger()
tprint("✅ [SIMPLE_LOGGER] Simple logger instance created")

# Mock the logger module structure
tprint("🔧 [SIMPLE_LOGGER] Setting up MockLogger class...")
class MockLogger:
    def __init__(self):
        self.tprint = _get_tprint()

    def getChild(self, name):
        self.tprint(f"🔧 [SIMPLE_LOGGER] Creating child logger: {name}")
        child = logging.getLogger(f'AresSimple.{name}')
        if not child.handlers:
            self.tprint(f"🔧 [SIMPLE_LOGGER] Adding console handler to child logger: {name}")
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            child.addHandler(console_handler)
        return child

    def info(self, msg):
        self.tprint(f"📝 [SIMPLE_LOGGER] INFO: {msg}")
        system_logger.info(msg)

    def warning(self, msg):
        self.tprint(f"⚠️ [SIMPLE_LOGGER] WARNING: {msg}")
        system_logger.warning(msg)

    def error(self, msg):
        self.tprint(f"❌ [SIMPLE_LOGGER] ERROR: {msg}")
        system_logger.error(msg)

    def debug(self, msg):
        self.tprint(f"🔍 [SIMPLE_LOGGER] DEBUG: {msg}")
        system_logger.debug(msg)

tprint("🔧 [SIMPLE_LOGGER] Attaching MockLogger to system_logger...")
mock_logger = MockLogger()
system_logger.getChild = mock_logger.getChild
tprint("✅ [SIMPLE_LOGGER] MockLogger attached successfully")

tprint("✅ [SIMPLE_LOGGER] Simple logger created successfully")
tprint("=" * 60)
