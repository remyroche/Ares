#!/usr/bin/env python3
"""
Wrapper script for Ares Launcher that ensures Poetry environment is used.
This ensures all dependencies (including VectorBT) are available.

Usage:
    python ares_launcher.py <args>
    
This will automatically run the actual launcher with the Poetry environment.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Path to the actual launcher
    launcher_path = project_root / "src" / "launcher" / "ares_launcher.py"
    
    if not launcher_path.exists():
        print(f"Error: Could not find launcher at {launcher_path}")
        sys.exit(1)
    
    # Build the command to run with poetry
    cmd = ["poetry", "run", "python", str(launcher_path)] + sys.argv[1:]
    
    # Print what we're running for transparency
    print(f"Running: {' '.join(cmd)}")
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, cwd=str(project_root))
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error running launcher: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

