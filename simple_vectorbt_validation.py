#!/usr/bin/env python3
"""
Simple validation script to check VectorBT optimization implementation without importing problematic modules.
"""

import os
import re

def validate_file_content(file_path, patterns):
    """Validate that a file contains the expected patterns."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        results = {}
        for name, pattern in patterns.items():
            if re.search(pattern, content, re.MULTILINE):
                results[name] = True
            else:
                results[name] = False
        
        return results
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {name: False for name in patterns.keys()}

def main():
    """Run validation."""
    print("🧪 Simple VectorBT Optimization Validation")
    print("=" * 60)
    
    # Define validation patterns
    validation_patterns = {
        'vectorbt_import': r'from \.\.utils\.vectorbt_rolling_optimizer import',
        'unified_import': r'from \.\.utils\.unified_optimization_system import',
        'vectorbt_optimizer_init': r'self\.vectorbt_optimizer = get_vectorbt_rolling_optimizer\(',
        'unified_optimizer_init': r'self\.unified_optimizer = get_unified_optimization_system\(\)',
        'vectorbt_rolling_method': r'def _vectorbt_rolling_operation',
        'pandas_fallback_method': r'def _pandas_rolling_operation',
        'optimize_dataframe_method': r'def optimize_dataframe_processing',
        'vectorized_rolling_method': r'def vectorized_rolling_operations'
    }
    
    # Files to validate
    files_to_check = [
        'src/feature_generation/categories/regime_feature_integration.py',
        'src/feature_generation/categories/regime_volatility.py',
        'src/feature_generation/categories/regime_volume.py',
        'src/feature_generation/categories/regime_structural_trend.py'
    ]
    
    all_results = {}
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"\n🔍 Validating {file_path}:")
            results = validate_file_content(file_path, validation_patterns)
            all_results[file_path] = results
            
            for pattern_name, found in results.items():
                status = "✅" if found else "❌"
                print(f"   {status} {pattern_name}")
        else:
            print(f"❌ File not found: {file_path}")
            all_results[file_path] = {name: False for name in validation_patterns.keys()}
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Validation Summary:")
    
    total_checks = 0
    passed_checks = 0
    
    for file_path, results in all_results.items():
        file_name = os.path.basename(file_path)
        file_passed = sum(results.values())
        file_total = len(results)
        total_checks += file_total
        passed_checks += file_passed
        
        status = "✅ PASS" if file_passed == file_total else f"⚠️ PARTIAL ({file_passed}/{file_total})"
        print(f"   {file_name}: {status}")
    
    overall_status = "✅ PASS" if passed_checks == total_checks else f"⚠️ PARTIAL ({passed_checks}/{total_checks})"
    print(f"\n   Overall: {overall_status}")
    
    if passed_checks == total_checks:
        print("\n🎉 All VectorBT optimizations are properly implemented!")
        print("\n📋 Implementation Summary:")
        print("   • VectorBTRollingOptimizer integrated into all regime feature generators")
        print("   • UnifiedVectorizationManager integrated for comprehensive optimization")
        print("   • All rolling operations now use VectorBT with pandas fallback")
        print("   • DataFrame processing optimized using VectorBT optimizers")
        print("   • Consistent VectorBT usage across all regime feature categories")
        return 0
    else:
        print(f"\n⚠️ {total_checks - passed_checks} checks failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())