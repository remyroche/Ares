#!/usr/bin/env python3
"""
Very simple test without logger
"""

print("Testing without logger...")

# Temporarily disable logger import
import sys
sys.modules['src.utils.logger'] = None

try:
    print("1. Testing launcher import...")
    from src.launcher.ares_launcher import AresLauncher
    print("   ✅ Launcher import OK")

    print("2. Testing launcher instantiation...")
    launcher = AresLauncher()
    print("   ✅ Launcher instantiation OK")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

