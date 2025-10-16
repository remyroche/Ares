"""
Verification Script for Ares Launcher Updates

This script verifies that all references to the legacy "pid_based_feature_generation"
stage have been properly updated to "interactive_feature_generation" in the
ares_launcher.py file and related configuration files.
"""

import os
import re
from pathlib import Path


def check_file_for_patterns(file_path: str, patterns: list, description: str) -> bool:
    """Check if a file contains specific patterns."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        print(f"\n🔍 Checking {description}:")
        all_found = True
        
        for pattern in patterns:
            if re.search(pattern, content):
                print(f"  ✅ Found: {pattern}")
            else:
                print(f"  ❌ Missing: {pattern}")
                all_found = False
        
        return all_found
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return False


def main():
    """Run verification checks for ares_launcher.py updates."""
    
    print("🔍 Ares Launcher Update Verification")
    print("=" * 50)
    
    # File path
    ares_launcher_file = Path(__file__).resolve().parent / "ares_launcher.py"
    
    # Check 1: Verify old references are removed
    print("\n📋 Check 1: Verifying old references are removed...")
    old_patterns = [
        r'pid_based_feature_generation',
        r'PID-based feature generation with interaction, polynomial, and cross-timeframe features',
        r'pid_based_features\\.parquet'
    ]
    
    try:
        with open(ares_launcher_file, 'r') as f:
            content = f.read()
        
        old_found = False
        for pattern in old_patterns:
            if re.search(pattern, content):
                print(f"  ❌ Found old reference: {pattern}")
                old_found = True
        
        if not old_found:
            print("  ✅ No old references found")
        else:
            print("  ⚠️ Old references still present - needs cleanup")
            
    except FileNotFoundError:
        print(f"❌ File not found: {ares_launcher_file}")
        return False
    
    # Check 2: Verify new references are present
    print("\n📋 Check 2: Verifying new references are present...")
    new_patterns = [
        r'interactive_feature_generation',
        r'Interactive feature generation with optimized lookbacks, cross-timeframe coverage, and matrix acceleration',
        r'features_<symbol>_<timeframe>\\.parquet'
    ]
    
    all_new_found = True
    for pattern in new_patterns:
        if re.search(pattern, content):
            print(f"  ✅ Found new reference: {pattern}")
        else:
            print(f"  ❌ Missing new reference: {pattern}")
            all_new_found = False
    
    # Check 3: Verify specific locations
    print("\n📋 Check 3: Verifying specific locations...")
    
    # Check sub_pipelines list
    if "'interactive_feature_generation'" in content:
        print("  ✅ Found in sub_pipelines list")
    else:
        print("  ❌ Missing from sub_pipelines list")
        all_new_found = False

    # Check description
    if "Interactive feature generation with optimized lookbacks, cross-timeframe coverage, and matrix acceleration" in content:
        print("  ✅ Found in description")
    else:
        print("  ❌ Missing from description")
        all_new_found = False

    # Check dependencies
    if "'interactive_feature_generation': ['feature_lookback_optimization']" in content:
        print("  ✅ Found in dependencies")
    else:
        print("  ❌ Missing from dependencies")
        all_new_found = False

    # Check outputs
    if "'interactive_feature_generation': [" in content and 'interactions_<symbol>_<timeframe>.parquet' in content:
        print("  ✅ Found in outputs")
    else:
        print("  ❌ Missing from outputs")
        all_new_found = False

    # Check 4: Verify migration config
    print("\n📋 Check 4: Verifying migration config...")
    migration_config_file = Path(__file__).resolve().parents[2] / "config" / "migration_config.yaml"
    
    try:
        with open(migration_config_file, 'r') as f:
            migration_content = f.read()
        
        if "interactive_feature_generation_component.py" in migration_content:
            print("  ✅ Found updated reference in migration config")
        else:
            print("  ❌ Missing updated reference in migration config")
            all_new_found = False

        if "pid_based_feature_generation_integration.py" in migration_content:
            print("  ❌ Found old reference in migration config")
            all_new_found = False
        else:
            print("  ✅ No old references in migration config")
            
    except FileNotFoundError:
        print(f"  ⚠️ Migration config file not found: {migration_config_file}")
    
    # Summary
    print("\n" + "=" * 50)
    if all_new_found and not old_found:
        print("🎉 All verification checks passed!")
        print("✅ Ares launcher has been successfully updated to use interactive feature generation")
        return True
    else:
        print("❌ Some verification checks failed")
        print("⚠️ Please review and fix the issues above")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)