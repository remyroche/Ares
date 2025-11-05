#!/usr/bin/env python3
"""
Example of calling Sticky Finite HMM Regime Discovery Step through ARES Launcher.
"""

import subprocess
import sys
from pathlib import Path

def run_with_launcher():
    """Run the step through ARES launcher."""
    
    print("🚀 Calling Sticky Finite HMM Regime Discovery Step via ARES Launcher")
    print("=" * 80)
    
    # Change to project root
    project_root = Path(__file__).parent
    launcher_path = project_root / "src" / "launcher" / "ares_launcher.py"
    
    if not launcher_path.exists():
        print(f"❌ ARES launcher not found at: {launcher_path}")
        return 1
    
    # Command to run the step
    cmd = [
        "python3", str(launcher_path),
        "--step", "sticky_finite_hmm_regime_discovery",
        "--symbol", "ETHUSDT",
        "--exchange", "binance", 
        "--timeframe", "1h",
        "--execution-mode", "light",
        "--enable-auto-tuning"
    ]
    
    print(f"📊 Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the launcher
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        
        print("📤 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("📥 STDERR:")
            print(result.stderr)
        
        print(f"🔚 Exit code: {result.returncode}")
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Failed to run launcher: {e}")
        return 1


if __name__ == "__main__":
    exit_code = run_with_launcher()
    sys.exit(exit_code)
