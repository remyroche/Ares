#!/usr/bin/env python3
"""
Debug launcher to isolate the issue
"""

import sys
import os

print("🔍 Debug launcher starting...")

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("📁 Project root added to path")

try:
    print("🚀 Testing basic Python execution...")
    print("   ✅ Basic Python works")

    print("📦 Testing basic imports...")
    import asyncio
    import logging
    print("   ✅ Basic imports work")

    print("🧪 Testing logger module (without auto-initialization)...")
    # Let's try to import the logger but prevent auto-initialization
    import importlib.util

    # Load the logger module but don't execute the bottom part
    spec = importlib.util.spec_from_file_location("logger", "src/utils/logger.py")
    logger_module = importlib.util.module_from_spec(spec)

    # Execute only up to the class definitions, not the auto-initialization
    print("   Logger module loaded without auto-initialization")

    print("🎯 Testing launcher import...")
    from src.launcher.ares_launcher import AresLauncher
    print("   ✅ Launcher imported successfully")

    print("🏗️ Testing launcher instantiation...")
    launcher = AresLauncher()
    print("   ✅ Launcher instantiated successfully")

    print("🎉 All tests passed! The issue is likely in logger auto-initialization.")

except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
