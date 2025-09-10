#!/usr/bin/env python3
"""
Simple test to isolate the import issue
"""

print("Starting test import...")

try:
    print("1. Testing basic imports...")
    import sys
    import os
    print("   Basic imports OK")

    print("2. Testing logger import...")
    from src.utils.logger import system_logger
    print("   Logger import OK")

    print("3. Testing launcher import...")
    from src.launcher.ares_launcher import AresLauncher
    print("   Launcher import OK")

    print("4. Testing launcher instantiation...")
    launcher = AresLauncher()
    print("   Launcher instantiation OK")

    print("✅ All imports successful!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

