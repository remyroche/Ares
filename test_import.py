#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Simple test to isolate the import issue
"""

tprint("Starting test import...")

try:
    tprint("1. Testing basic imports...")
    import sys
    import os
    tprint("   Basic imports OK")

    tprint("2. Testing logger import...")
    from src.utils.logger import system_logger
    tprint("   Logger import OK")

    tprint("3. Testing launcher import...")
    from src.launcher.ares_launcher import AresLauncher
    tprint("   Launcher import OK")

    tprint("4. Testing launcher instantiation...")
    launcher = AresLauncher()
    tprint("   Launcher instantiation OK")

    tprint("✅ All imports successful!")

except Exception as e:
    tprint(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

