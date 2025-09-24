#!/usr/bin/env python3
"""
Verification Script for Timeframe Changes

This script verifies that all timeframe changes are consistent across the system:
- Analyst: 15m base timeframe
- Tactician: 5m base timeframe  
- Regime Detection: 1h base timeframe
"""

import os
import sys
from pathlib import Path

def verify_analyst_timeframes():
    """Verify Analyst model uses 15m base timeframe."""
    print("🔍 Verifying Analyst Model Timeframes (15m base)...")
    
    try:
        with open("src/training/steps/market_analysis/automatic_timeframe_optimizer.py", 'r') as f:
            content = f.read()
        
        # Check for 15m base timeframe references
        checks = [
            "15m base timeframe",
            "15 minutes (1 * 15m)",
            "120 minutes (8 * 15m)",
            "30m and 60m"
        ]
        
        all_found = True
        for check in checks:
            if check in content:
                print(f"   ✅ {check}")
            else:
                print(f"   ❌ {check} - MISSING")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False


def verify_tactician_timeframes():
    """Verify Tactician model uses 5m base timeframe."""
    print("\n🔍 Verifying Tactician Model Timeframes (5m base)...")
    
    try:
        with open("src/training/steps/market_analysis/automatic_timeframe_optimizer.py", 'r') as f:
            content = f.read()
        
        # Check for 5m base timeframe references
        checks = [
            "5m base timeframe",
            "20 minutes (4 * 5m)",
            "80 minutes (16 * 5m)",
            "20m and 40m"
        ]
        
        all_found = True
        for check in checks:
            if check in content:
                print(f"   ✅ {check}")
            else:
                print(f"   ❌ {check} - MISSING")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False


def verify_regime_detection_timeframes():
    """Verify regime detection uses 1h base timeframe."""
    print("\n🔍 Verifying Regime Detection Timeframes (1h base)...")
    
    try:
        with open("src/training/steps/market_analysis/tas_regime/core/tas_regime_config.py", 'r') as f:
            content = f.read()
        
        # Check for 1h base timeframe references
        checks = [
            'regime_detection_timeframe: str = "1h"',
            '["1m", "5m", "15m", "1h"]'
        ]
        
        all_found = True
        for check in checks:
            if check in content:
                print(f"   ✅ {check}")
            else:
                print(f"   ❌ {check} - MISSING")
                all_found = False
        
        # Check data ingestion
        with open("src/training/steps/market_analysis/tas_regime/data_pipeline/data_ingestion.py", 'r') as f:
            content = f.read()
        
        if 'timeframe: str = "1h"  # 1h base timeframe for regime detection' in content:
            print("   ✅ Data ingestion timeframe updated")
        else:
            print("   ❌ Data ingestion timeframe not updated")
            all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"   ❌ Error reading files: {e}")
        return False


def verify_optimization_configs():
    """Verify optimization configurations are updated."""
    print("\n🔍 Verifying Optimization Configurations...")
    
    try:
        with open("src/training/steps/market_analysis/automatic_timeframe_optimizer.py", 'r') as f:
            content = f.read()
        
        # Check for updated optimization configs
        checks = [
            "Analyst-specific optimization config (15m base timeframe)",
            "Tactician-specific optimization config (5m base timeframe)",
            "min_horizon=1,  # 15 minutes (1 * 15m)",
            "min_horizon=4,   # 20 minutes (4 * 5m)"
        ]
        
        all_found = True
        for check in checks:
            if check in content:
                print(f"   ✅ {check}")
            else:
                print(f"   ❌ {check} - MISSING")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False


def verify_fallback_configs():
    """Verify fallback configurations are updated."""
    print("\n🔍 Verifying Fallback Configurations...")
    
    try:
        with open("src/training/steps/market_analysis/enhanced_multi_horizon_pipeline.py", 'r') as f:
            content = f.read()
        
        # Check for updated fallback configs
        checks = [
            "Analyst fallback: 15m base timeframe (2 periods = 30m, 4 periods = 60m)",
            "Tactician fallback: 5m base timeframe (4 periods = 20m, 8 periods = 40m)",
            "30m and 60m",
            "20m and 40m"
        ]
        
        all_found = True
        for check in checks:
            if check in content:
                print(f"   ✅ {check}")
            else:
                print(f"   ❌ {check} - MISSING")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False


def verify_multi_horizon_config():
    """Verify multi-horizon configuration is updated."""
    print("\n🔍 Verifying Multi-Horizon Configuration...")
    
    try:
        with open("src/training/steps/market_analysis/multi_horizon_profit_labeler.py", 'r') as f:
            content = f.read()
        
        # Check for updated multi-horizon config
        checks = [
            "UPDATED for new base timeframes",
            "for Tactician 5m base",
            "10 minutes (2 * 5m) - for Tactician 5m base",
            "20 minutes (4 * 5m) - for Tactician 5m base"
        ]
        
        all_found = True
        for check in checks:
            if check in content:
                print(f"   ✅ {check}")
            else:
                print(f"   ❌ {check} - MISSING")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False


def main():
    """Run all timeframe verification checks."""
    print("🚀 Verifying Timeframe Changes")
    print("=" * 50)
    print("Expected Changes:")
    print("  • Analyst: 15m base timeframe")
    print("  • Tactician: 5m base timeframe")
    print("  • Regime Detection: 1h base timeframe")
    print("=" * 50)
    
    checks = [
        ("Analyst Timeframes", verify_analyst_timeframes),
        ("Tactician Timeframes", verify_tactician_timeframes),
        ("Regime Detection Timeframes", verify_regime_detection_timeframes),
        ("Optimization Configurations", verify_optimization_configs),
        ("Fallback Configurations", verify_fallback_configs),
        ("Multi-Horizon Configuration", verify_multi_horizon_config)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
            if result:
                print(f"✅ {check_name} PASSED")
            else:
                print(f"❌ {check_name} FAILED")
        except Exception as e:
            print(f"❌ {check_name} FAILED with exception: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("📊 VERIFICATION SUMMARY")
    print("="*50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {check_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All timeframe changes verified successfully!")
        print("\n📋 TIMEFRAME SUMMARY:")
        print("   ✅ Analyst Model: 15m base timeframe")
        print("     → Immediate: 2 periods = 30 minutes")
        print("     → Short: 4 periods = 60 minutes")
        print("   ✅ Tactician Model: 5m base timeframe")
        print("     → Immediate: 4 periods = 20 minutes")
        print("     → Short: 8 periods = 40 minutes")
        print("   ✅ Regime Detection: 1h base timeframe")
        print("     → Detection timeframe: 1h")
        print("     → Trading timeframes: 1m, 5m, 15m, 1h")
    else:
        print("\n⚠️ Some timeframe changes need attention.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
