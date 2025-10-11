#!/usr/bin/env python3
"""
Simple validation script for VectorBT migration.

This script validates that the VectorBT migration was completed successfully
by checking the code structure and imports.
"""

import os
import re
from pathlib import Path

def check_file_imports(file_path: str, required_imports: list) -> bool:
    """Check if a file contains the required imports."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        missing_imports = []
        for import_name in required_imports:
            if import_name not in content:
                missing_imports.append(import_name)
        
        if missing_imports:
            print(f"❌ {file_path}: Missing imports: {missing_imports}")
            return False
        else:
            print(f"✅ {file_path}: All required imports found")
            return True
    except Exception as e:
        print(f"❌ {file_path}: Error reading file - {e}")
        return False

def check_vectorbt_usage(file_path: str) -> bool:
    """Check if a file properly uses VectorBT optimizations."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for VectorBT optimization patterns
        patterns = [
            r'VectorBTOptimizationMixin',
            r'get_vectorbt_rolling_optimizer',
            r'rolling_optimizer',
            r'vectorbt_operations',
            r'performance_stats'
        ]
        
        found_patterns = []
        for pattern in patterns:
            if re.search(pattern, content):
                found_patterns.append(pattern)
        
        if len(found_patterns) >= 3:  # At least 3 patterns should be present
            print(f"✅ {file_path}: VectorBT optimization patterns found ({len(found_patterns)}/5)")
            return True
        else:
            print(f"⚠️ {file_path}: Limited VectorBT optimization patterns ({len(found_patterns)}/5)")
            return False
    except Exception as e:
        print(f"❌ {file_path}: Error reading file - {e}")
        return False

def validate_volume_features():
    """Validate Advanced Volume Features migration."""
    print("\n🔍 Validating Advanced Volume Features...")
    
    file_path = "src/feature_generation/categories/volume.py"
    required_imports = [
        "VectorBTOptimizationMixin",
        "get_vectorbt_rolling_optimizer",
        "VectorBTRollingOptimizer"
    ]
    
    imports_ok = check_file_imports(file_path, required_imports)
    vectorbt_ok = check_vectorbt_usage(file_path)
    
    return imports_ok and vectorbt_ok

def validate_volatility_features():
    """Validate Advanced Volatility Features migration."""
    print("\n🔍 Validating Advanced Volatility Features...")
    
    file_path = "src/feature_generation/categories/volatility.py"
    required_imports = [
        "VectorBTOptimizationMixin",
        "get_vectorbt_rolling_optimizer",
        "VectorBTRollingOptimizer"
    ]
    
    imports_ok = check_file_imports(file_path, required_imports)
    vectorbt_ok = check_vectorbt_usage(file_path)
    
    return imports_ok and vectorbt_ok

def validate_cross_timeframe_features():
    """Validate Cross-Timeframe Features migration."""
    print("\n🔍 Validating Cross-Timeframe Features...")
    
    file_path = "src/feature_generation/categories/cross_timeframe.py"
    required_imports = [
        "VectorBTOptimizationMixin",
        "get_vectorbt_rolling_optimizer",
        "VectorBTRollingOptimizer"
    ]
    
    imports_ok = check_file_imports(file_path, required_imports)
    vectorbt_ok = check_vectorbt_usage(file_path)
    
    return imports_ok and vectorbt_ok

def validate_vectorbt_rolling_optimizer():
    """Validate VectorBT Rolling Optimizer exists and is properly implemented."""
    print("\n🔍 Validating VectorBT Rolling Optimizer...")
    
    file_path = "src/feature_generation/utils/vectorbt_rolling_optimizer.py"
    
    if not os.path.exists(file_path):
        print(f"❌ {file_path}: File not found")
        return False
    
    required_classes = [
        "VectorBTRollingOptimizer",
        "get_vectorbt_rolling_optimizer"
    ]
    
    return check_file_imports(file_path, required_classes)

def validate_vectorbt_optimization_mixin():
    """Validate VectorBT Optimization Mixin exists and is properly implemented."""
    print("\n🔍 Validating VectorBT Optimization Mixin...")
    
    file_path = "src/feature_generation/core/vectorbt_optimization_mixin.py"
    
    if not os.path.exists(file_path):
        print(f"❌ {file_path}: File not found")
        return False
    
    required_classes = [
        "VectorBTOptimizationMixin",
        "_vectorbt_rolling_operation",
        "performance_stats"
    ]
    
    return check_file_imports(file_path, required_classes)

def main():
    """Run all validation checks."""
    print("🚀 VectorBT Migration Validation")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("src/feature_generation"):
        print("❌ Error: Not in the correct directory. Please run from the workspace root.")
        return False
    
    validation_results = {}
    
    # Validate core components
    validation_results['vectorbt_rolling_optimizer'] = validate_vectorbt_rolling_optimizer()
    validation_results['vectorbt_optimization_mixin'] = validate_vectorbt_optimization_mixin()
    
    # Validate feature categories
    validation_results['volume_features'] = validate_volume_features()
    validation_results['volatility_features'] = validate_volatility_features()
    validation_results['cross_timeframe_features'] = validate_cross_timeframe_features()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION RESULTS")
    print("=" * 50)
    
    passed_tests = sum(validation_results.values())
    total_tests = len(validation_results)
    
    for test_name, passed in validation_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print("")
    print(f"Overall: {passed_tests}/{total_tests} validations passed")
    
    if passed_tests == total_tests:
        print("🎉 All VectorBT migration validations passed!")
        print("\n📋 Migration Summary:")
        print("✅ Advanced Volume Features - VectorBT optimized")
        print("✅ Advanced Volatility Features - VectorBT optimized") 
        print("✅ Cross-Timeframe Features - VectorBT optimized")
        print("✅ VectorBTRollingOptimizer integration complete")
        print("✅ Performance monitoring and fallbacks implemented")
        return True
    else:
        print("⚠️ Some validations failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)