#!/usr/bin/env python3
"""
Terminal Diagnostic Script
Helps diagnose terminal and system issues
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return the result"""
    try:
        print(f"🔧 Testing: {description}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ {description}: SUCCESS")
            return True
        else:
            print(f"❌ {description}: FAILED")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description}: TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description}: EXCEPTION - {e}")
        return False

def main():
    print("🔍 Terminal Diagnostic Starting...")
    print("=" * 50)

    # Test basic terminal functionality
    tests = [
        ("echo 'Hello World'", "Basic echo command"),
        ("pwd", "Print working directory"),
        ("which python3", "Check Python3 availability"),
        ("python3 --version", "Check Python3 version"),
        ("ps aux | grep python | wc -l", "Count Python processes"),
        ("df -h /", "Check disk space"),
        ("free -h", "Check memory usage"),
    ]

    passed = 0
    for cmd, desc in tests:
        if run_command(cmd, desc):
            passed += 1
        print()

    print("=" * 50)
    print(f"Diagnostic Complete: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("✅ All tests passed - terminal appears functional")
    else:
        print("❌ Some tests failed - terminal may have issues")

    # Try to run the Ares launcher with minimal options
    print("\n🔧 Testing Ares launcher...")
    try:
        # Change to project directory
        os.chdir("/Users/remyroche/Documents/Ares")

        # Try importing the launcher module first
        print("Testing launcher import...")
        sys.path.insert(0, "/Users/remyroche/Documents/Ares")
        import src.launcher.ares_launcher as launcher
        print("✅ Launcher module imported successfully")

        # Try running with --help
        print("Testing launcher --help...")
        result = subprocess.run(["python3", "src/launcher/ares_launcher.py", "--help"],
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Launcher --help works")
        else:
            print(f"❌ Launcher --help failed: {result.stderr}")

    except Exception as e:
        print(f"❌ Launcher test failed: {e}")

if __name__ == "__main__":
    main()

