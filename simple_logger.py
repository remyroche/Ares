#!/usr/bin/env python3
"""
Simple logger replacement to bypass initialization issues
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Create a simple logger that works
def create_simple_logger():
    """Create a basic logger without complex initialization."""
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
print("🔧 [SIMPLE_LOGGER] Creating simple logger instance...")
system_logger = create_simple_logger()
print("✅ [SIMPLE_LOGGER] Simple logger instance created")

# Mock the logger module structure
print("🔧 [SIMPLE_LOGGER] Setting up MockLogger class...")
class MockLogger:
    def getChild(self, name):
        print(f"🔧 [SIMPLE_LOGGER] Creating child logger: {name}")
        child = logging.getLogger(f'AresSimple.{name}')
        if not child.handlers:
            print(f"🔧 [SIMPLE_LOGGER] Adding console handler to child logger: {name}")
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            child.addHandler(console_handler)
        return child

    def info(self, msg):
        print(f"📝 [SIMPLE_LOGGER] INFO: {msg}")
        system_logger.info(msg)

    def warning(self, msg):
        print(f"⚠️ [SIMPLE_LOGGER] WARNING: {msg}")
        system_logger.warning(msg)

    def error(self, msg):
        print(f"❌ [SIMPLE_LOGGER] ERROR: {msg}")
        system_logger.error(msg)

    def debug(self, msg):
        print(f"🔍 [SIMPLE_LOGGER] DEBUG: {msg}")
        system_logger.debug(msg)

print("🔧 [SIMPLE_LOGGER] Attaching MockLogger to system_logger...")
system_logger.getChild = MockLogger().getChild
print("✅ [SIMPLE_LOGGER] MockLogger attached successfully")

print("✅ [SIMPLE_LOGGER] Simple logger created successfully")
print("=" * 60)
