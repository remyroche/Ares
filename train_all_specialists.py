#!/usr/bin/env python3
"""
Script to train all 11 specialist models manually.
"""

import subprocess
import sys
import os

# List of all specialist steps to train
SPECIALISTS = [
    "enhanced_ml_momentum_persistence_step",
    "enhanced_ml_smc_regime_step",
    "enhanced_ml_volatility_burst_step",
    "enhanced_ml_volume_force_step",
    "enhanced_xgb_macro_regime_step",
    "enhanced_xgb_meso_regime_step",
    "enhanced_ml_liquidity_regime_step",
    "enhanced_ml_path_regime_step",
    "enhanced_ml_risk_regime_step",
    "enhanced_ml_microstructure_step",
    "enhanced_ml_spectral_step"
]

def run_specialist(specialist_name):
    """Run a single specialist training step."""
    print(f"\n🚀 Training {specialist_name}...")

    cmd = [
        "poetry", "run", "python3", "src/launcher/ares_launcher.py",
        specialist_name, "--symbol", "ETHUSDT", "--execution-mode", "full"
    ]

    try:
        result = subprocess.run(cmd, cwd="/Users/remyroche/Documents/Ares",
                              capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"✅ {specialist_name} completed successfully")
            return True
        else:
            print(f"❌ {specialist_name} failed with exit code {result.returncode}")
            print(f"STDERR: {result.stderr[-500:]}")  # Last 500 chars of error
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {specialist_name} timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"💥 {specialist_name} failed with exception: {e}")
        return False

def main():
    """Train all specialists."""
    print("🎯 Training all 11 specialist models...")
    print("=" * 50)

    successful = 0
    failed = 0

    for specialist in SPECIALISTS:
        if run_specialist(specialist):
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 50)
    print(f"📊 Training Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {successful}/{len(SPECIALISTS)} ({successful/len(SPECIALISTS)*100:.1f}%)")

    if failed == 0:
        print("\n🎉 All specialists trained successfully!")
        return 0
    else:
        print(f"\n⚠️ {failed} specialists failed to train")
        return 1

if __name__ == "__main__":
    sys.exit(main())