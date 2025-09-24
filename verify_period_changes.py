#!/usr/bin/env python3
"""
Verification Script for Period Changes (1-16 periods)

This script verifies that all period changes are consistent across the system:
- Analyst: 1-16 periods (15m base = 15m-240m)
- Tactician: 1-16 periods (5m base = 5m-80m)
"""

import os
import sys
from pathlib import Path

def verify_analyst_periods():
    """Verify Analyst model uses 1-16 periods."""
    print("🔍 Verifying Analyst Model Periods (1-16 periods, 15m base)...")
    
    try:
        with open("src/training/steps/market_analysis/automatic_timeframe_optimizer.py", 'r') as f:
            content = f.read()
        
        # Check for 1-16 period references
        checks = [
            "Analyst-specific optimization config (15m base timeframe, 1-16 periods)",
            "min_horizon=1,  # 15 minutes (1 * 15m)",
            "max_horizon=16,  # 240 minutes (16 * 15m)",
            "1-16 periods = 15m-240m",
            "30m and 120m"
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


def verify_tactician_periods():
    """Verify Tactician model uses 1-16 periods."""
    print("\n🔍 Verifying Tactician Model Periods (1-16 periods, 5m base)...")
    
    try:
        with open("src/training/steps/market_analysis/automatic_timeframe_optimizer.py", 'r') as f:
            content = f.read()
        
        # Check for 1-16 period references
        checks = [
            "Tactician-specific optimization config (5m base timeframe, 1-16 periods)",
            "min_horizon=1,   # 5 minutes (1 * 5m)",
            "max_horizon=16,  # 80 minutes (16 * 5m)",
            "1-16 periods = 5m-80m",
            "10m and 40m"
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


def verify_fallback_configurations():
    """Verify fallback configurations use 1-16 periods."""
    print("\n🔍 Verifying Fallback Configurations...")
    
    try:
        with open("src/training/steps/market_analysis/automatic_timeframe_optimizer.py", 'r') as f:
            content = f.read()
        
        # Check for updated fallback configs
        checks = [
            "Analyst: 15m base timeframe (1-16 periods = 15m-240m)",
            "Tactician: 5m base timeframe (1-16 periods = 5m-80m)",
            "immediate': 2, 'short': 8}  # 30m and 120m",
            "immediate': 2, 'short': 8}  # 10m and 40m"
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


def verify_enhanced_pipeline_configs():
    """Verify enhanced pipeline configurations."""
    print("\n🔍 Verifying Enhanced Pipeline Configurations...")
    
    try:
        with open("src/training/steps/market_analysis/enhanced_multi_horizon_pipeline.py", 'r') as f:
            content = f.read()
        
        # Check for updated enhanced pipeline configs
        checks = [
            "Analyst fallback: 15m base timeframe (1-16 periods = 15m-240m)",
            "Tactician fallback: 5m base timeframe (1-16 periods = 5m-80m)",
            "30m and 120m",
            "10m and 40m"
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


def verify_optimization_parameters():
    """Verify optimization parameters are updated."""
    print("\n🔍 Verifying Optimization Parameters...")
    
    try:
        with open("src/training/steps/market_analysis/automatic_timeframe_optimizer.py", 'r') as f:
            content = f.read()
        
        # Check for updated optimization parameters
        checks = [
            "horizon_step=1,  # Test every period from 1-16",
            "n_target_candidates=8,  # Increased for 1-16 periods",
            "bayesian_iterations=25  # Analyst optimization",
            "bayesian_iterations=30  # Tactician optimization"
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


def verify_combined_configurations():
    """Verify combined configurations for both models."""
    print("\n🔍 Verifying Combined Configurations...")
    
    try:
        with open("src/training/steps/market_analysis/automatic_timeframe_optimizer.py", 'r') as f:
            content = f.read()
        
        # Check for combined configurations
        checks = [
            "Balanced approach",
            "immediate': 2, 'short': 8}  # Balanced approach"
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
    """Run all period verification checks."""
    print("🚀 Verifying Period Changes (1-16 periods)")
    print("=" * 60)
    print("Expected Changes:")
    print("  • Analyst: 1-16 periods (15m base = 15m-240m)")
    print("  • Tactician: 1-16 periods (5m base = 5m-80m)")
    print("  • Both models: horizon_step=1 for full exploration")
    print("=" * 60)
    
    checks = [
        ("Analyst Periods", verify_analyst_periods),
        ("Tactician Periods", verify_tactician_periods),
        ("Fallback Configurations", verify_fallback_configurations),
        ("Enhanced Pipeline Configs", verify_enhanced_pipeline_configs),
        ("Optimization Parameters", verify_optimization_parameters),
        ("Combined Configurations", verify_combined_configurations)
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
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {check_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All period changes verified successfully!")
        print("\n📋 PERIOD SUMMARY:")
        print("   ✅ Analyst Model: 1-16 periods (15m base)")
        print("     → Range: 15 minutes to 240 minutes")
        print("     → Optimization: Full exploration with horizon_step=1")
        print("     → Fallback: 2 periods (30m) and 8 periods (120m)")
        print("   ✅ Tactician Model: 1-16 periods (5m base)")
        print("     → Range: 5 minutes to 80 minutes")
        print("     → Optimization: Full exploration with horizon_step=1")
        print("     → Fallback: 2 periods (10m) and 8 periods (40m)")
        print("   ✅ Combined Models: Balanced approach")
        print("     → Both models: 2 periods (immediate) and 8 periods (short)")
    else:
        print("\n⚠️ Some period changes need attention.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
