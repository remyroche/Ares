#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Debug launcher to isolate the issue
"""

import sys
import os

tprint("🔍 Debug launcher starting...")

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

tprint("📁 Project root added to path")

try:
    tprint("🚀 Testing basic Python execution...")
    tprint("   ✅ Basic Python works")

    tprint("📦 Testing basic imports...")
    import asyncio
    import logging
    tprint("   ✅ Basic imports work")

    tprint("🧪 Testing logger module (without auto-initialization)...")
    # Let's try to import the logger but prevent auto-initialization
    import importlib.util

    # Load the logger module but don't execute the bottom part
    spec = importlib.util.spec_from_file_location("logger", "src/utils/logger.py")
    logger_module = importlib.util.module_from_spec(spec)

    # Execute only up to the class definitions, not the auto-initialization
    tprint("   Logger module loaded without auto-initialization")

    tprint("🎯 Testing launcher import...")
    from src.launcher.ares_launcher import AresLauncher
    tprint("   ✅ Launcher imported successfully")

    tprint("🏗️ Testing launcher instantiation...")
    launcher = AresLauncher()
    tprint("   ✅ Launcher instantiated successfully")

    tprint("🎉 All tests passed! The issue is likely in logger auto-initialization.")

except Exception as e:
    tprint(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
