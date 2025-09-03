#!/usr/bin/env python3
"""
Simple test runner that handles missing dependencies gracefully.
"""

import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    return missing_deps

def install_dependencies(deps):
    """Try to install missing dependencies."""
    print(f"Missing dependencies: {', '.join(deps)}")
    print("Attempting to install with pip...")
    
    for dep in deps:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--user", dep], check=True)
            print(f"✅ Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {dep}")
            return False
    return True

def run_tests():
    """Run the common operations tests."""
    # Check dependencies
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠️  Missing dependencies detected: {', '.join(missing)}")
        print("\nThe tests require numpy and pandas to run properly.")
        print("\nOptions:")
        print("1. Install dependencies manually in a virtual environment")
        print("2. Run with --install flag to attempt user installation")
        print("3. Run tests that don't require these dependencies")
        
        if "--install" in sys.argv:
            if install_dependencies(missing):
                print("\nDependencies installed. Running tests...")
            else:
                print("\nFailed to install dependencies.")
                return 1
        else:
            print("\nSkipping tests due to missing dependencies.")
            print("Run with --install flag to attempt installation.")
            return 1
    
    # Run the actual test script
    test_runner = Path(__file__).parent / "run_common_operations_tests.py"
    result = subprocess.run([sys.executable, str(test_runner)], capture_output=False)
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())