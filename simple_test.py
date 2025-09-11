#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Very simple test without logger
"""

tprint("Testing without logger...")

# Temporarily disable logger import
import sys
sys.modules['src.utils.logger'] = None

try:
    tprint("1. Testing launcher import...")
    from src.launcher.ares_launcher import AresLauncher
    tprint("   ✅ Launcher import OK")

    tprint("2. Testing launcher instantiation...")
    launcher = AresLauncher()
    tprint("   ✅ Launcher instantiation OK")

except Exception as e:
    tprint(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

