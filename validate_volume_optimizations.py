#!/usr/bin/env python3
"""
Simple validation script for volume feature VectorBT optimizations.

This script validates that the optimization code has been properly integrated
without requiring external dependencies.
"""

import os
import sys
import re

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
        print(f"❌ {file_path}: Error reading file: {e}")
        return False

def check_class_usage(file_path: str, class_name: str) -> bool:
    """Check if a class is being used in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        if class_name in content:
            print(f"✅ {file_path}: {class_name} usage found")
            return True
        else:
            print(f"❌ {file_path}: {class_name} usage not found")
            return False
    except Exception as e:
        print(f"❌ {file_path}: Error reading file: {e}")
        return False

def check_method_implementation(file_path: str, method_name: str) -> bool:
    """Check if a method is implemented in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Look for method definition
        pattern = rf'def\s+{method_name}\s*\('
        if re.search(pattern, content):
            print(f"✅ {file_path}: {method_name} method found")
            return True
        else:
            print(f"❌ {file_path}: {method_name} method not found")
            return False
    except Exception as e:
        print(f"❌ {file_path}: Error reading file: {e}")
        return False

def validate_volume_optimizations():
    """Validate VectorBT optimizations in volume features."""
    print("🔍 Validating VectorBT optimizations in volume features...\n")
    
    # Define file paths
    volume_file = "src/feature_generation/categories/volume.py"
    regime_volume_file = "src/feature_generation/categories/regime_volume.py"
    
    # Check if files exist
    if not os.path.exists(volume_file):
        print(f"❌ Volume file not found: {volume_file}")
        return False
    
    if not os.path.exists(regime_volume_file):
        print(f"❌ Regime volume file not found: {regime_volume_file}")
        return False
    
    print("✅ Required files found\n")
    
    # Validate volume.py optimizations
    print("📊 Validating volume.py optimizations...")
    
    # Check imports
    volume_imports = [
        "UnifiedVectorizationManager",
        "get_unified_vectorization_manager",
        "VectorBTRollingOptimizer",
        "get_vectorbt_rolling_optimizer"
    ]
    
    volume_imports_ok = check_file_imports(volume_file, volume_imports)
    
    # Check class usage
    volume_class_usage = [
        "UnifiedVectorizationManager",
        "VectorBTRollingOptimizer"
    ]
    
    volume_class_usage_ok = all(check_class_usage(volume_file, cls) for cls in volume_class_usage)
    
    # Check method implementation
    volume_methods = [
        "_optimized_rolling_operation"
    ]
    
    volume_methods_ok = all(check_method_implementation(volume_file, method) for method in volume_methods)
    
    print()
    
    # Validate regime_volume.py optimizations
    print("📊 Validating regime_volume.py optimizations...")
    
    # Check imports
    regime_imports = [
        "UnifiedVectorizationManager",
        "get_unified_vectorization_manager",
        "VectorBTRollingOptimizer",
        "get_vectorbt_rolling_optimizer"
    ]
    
    regime_imports_ok = check_file_imports(regime_volume_file, regime_imports)
    
    # Check class usage
    regime_class_usage = [
        "UnifiedVectorizationManager",
        "VectorBTRollingOptimizer"
    ]
    
    regime_class_usage_ok = all(check_class_usage(regime_volume_file, cls) for cls in regime_class_usage)
    
    # Check method updates
    regime_methods = [
        "_rolling_mean",
        "_rolling_std",
        "_vectorbt_rolling_operation"
    ]
    
    regime_methods_ok = all(check_method_implementation(regime_volume_file, method) for method in regime_methods)
    
    print()
    
    # Summary
    print("📋 Validation Summary:")
    print(f"  Volume file imports: {'✅' if volume_imports_ok else '❌'}")
    print(f"  Volume file class usage: {'✅' if volume_class_usage_ok else '❌'}")
    print(f"  Volume file methods: {'✅' if volume_methods_ok else '❌'}")
    print(f"  Regime volume file imports: {'✅' if regime_imports_ok else '❌'}")
    print(f"  Regime volume file class usage: {'✅' if regime_class_usage_ok else '❌'}")
    print(f"  Regime volume file methods: {'✅' if regime_methods_ok else '❌'}")
    
    all_checks = all([
        volume_imports_ok, volume_class_usage_ok, volume_methods_ok,
        regime_imports_ok, regime_class_usage_ok, regime_methods_ok
    ])
    
    if all_checks:
        print("\n🎉 All VectorBT optimizations have been successfully integrated!")
        return True
    else:
        print("\n❌ Some optimizations are missing. Please check the details above.")
        return False

def check_optimization_patterns():
    """Check for specific optimization patterns in the code."""
    print("\n🔍 Checking optimization patterns...\n")
    
    volume_file = "src/feature_generation/categories/volume.py"
    
    try:
        with open(volume_file, 'r') as f:
            content = f.read()
        
        # Check for optimization patterns
        patterns = [
            (r'unified_manager.*get_unified_vectorization_manager', "Unified Vectorization Manager initialization"),
            (r'rolling_optimizer.*get_vectorbt_rolling_optimizer', "VectorBT Rolling Optimizer initialization"),
            (r'def _optimized_rolling_operation', "Optimized rolling operation method"),
            (r'performance_stats.*vectorbt_operations', "Performance tracking"),
            (r'performance_stats.*unified_optimizations', "Unified optimization tracking"),
            (r'fallback.*pandas', "Intelligent fallback mechanism"),
            (r'VectorBT.*rolling.*optimizer', "VectorBT rolling optimizer usage")
        ]
        
        found_patterns = 0
        for pattern, description in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                print(f"✅ {description}")
                found_patterns += 1
            else:
                print(f"❌ {description}")
        
        print(f"\n📊 Found {found_patterns}/{len(patterns)} optimization patterns")
        return found_patterns >= len(patterns) * 0.8  # 80% threshold
        
    except Exception as e:
        print(f"❌ Error checking optimization patterns: {e}")
        return False

def main():
    """Main validation function."""
    print("🚀 VectorBT Volume Features Optimization Validation")
    print("=" * 60)
    
    # Run validations
    basic_validation = validate_volume_optimizations()
    pattern_validation = check_optimization_patterns()
    
    print("\n" + "=" * 60)
    print("📋 Final Results:")
    print(f"  Basic validation: {'✅ PASSED' if basic_validation else '❌ FAILED'}")
    print(f"  Pattern validation: {'✅ PASSED' if pattern_validation else '❌ FAILED'}")
    
    if basic_validation and pattern_validation:
        print("\n🎉 All validations passed! VectorBT optimizations are properly integrated.")
        return True
    else:
        print("\n❌ Some validations failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)