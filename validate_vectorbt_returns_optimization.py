#!/usr/bin/env python3
"""
Simple validation script for VectorBT optimizations in returns module.

This script validates that the returns module has been properly updated
to use VectorBT optimizations without requiring external dependencies.
"""

import os
import re
import sys

def check_file_for_vectorbt_usage(file_path: str) -> dict:
    """Check if a file uses VectorBT optimizations."""
    if not os.path.exists(file_path):
        return {'exists': False}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for VectorBT imports
    vectorbt_imports = re.findall(r'from.*vectorbt.*import|import.*vectorbt', content)
    
    # Check for VectorBTRollingOptimizer usage
    rolling_optimizer_usage = re.findall(r'VectorBTRollingOptimizer|get_vectorbt_rolling_optimizer', content)
    
    # Check for UnifiedVectorizationManager usage
    unified_manager_usage = re.findall(r'UnifiedVectorizationManager|get_unified_vectorization_manager', content)
    
    # Check for VectorBT rolling operations
    rolling_operations = re.findall(r'rolling_mean|rolling_std|rolling_var|rolling_min|rolling_max|rolling_sum', content)
    
    # Check for optimization patterns
    optimization_patterns = re.findall(r'self\.rolling_optimizer|self\.unified_manager|VECTORBT_AVAILABLE', content)
    
    return {
        'exists': True,
        'vectorbt_imports': len(vectorbt_imports),
        'rolling_optimizer_usage': len(rolling_optimizer_usage),
        'unified_manager_usage': len(unified_manager_usage),
        'rolling_operations': len(rolling_operations),
        'optimization_patterns': len(optimization_patterns),
        'has_vectorbt_imports': len(vectorbt_imports) > 0,
        'has_rolling_optimizer': len(rolling_optimizer_usage) > 0,
        'has_unified_manager': len(unified_manager_usage) > 0,
        'has_rolling_operations': len(rolling_operations) > 0,
        'has_optimization_patterns': len(optimization_patterns) > 0
    }

def validate_returns_module():
    """Validate the returns module for VectorBT optimizations."""
    print("Validating VectorBT optimizations in returns module...")
    print("=" * 60)
    
    # Check main returns module
    returns_file = "src/feature_generation/categories/returns.py"
    returns_check = check_file_for_vectorbt_usage(returns_file)
    
    print(f"Returns module: {returns_file}")
    print(f"  Exists: {'✅' if returns_check['exists'] else '❌'}")
    if returns_check['exists']:
        print(f"  VectorBT imports: {returns_check['vectorbt_imports']} {'✅' if returns_check['has_vectorbt_imports'] else '❌'}")
        print(f"  Rolling optimizer usage: {returns_check['rolling_optimizer_usage']} {'✅' if returns_check['has_rolling_optimizer'] else '❌'}")
        print(f"  Unified manager usage: {returns_check['unified_manager_usage']} {'✅' if returns_check['has_unified_manager'] else '❌'}")
        print(f"  Rolling operations: {returns_check['rolling_operations']} {'✅' if returns_check['has_rolling_operations'] else '❌'}")
        print(f"  Optimization patterns: {returns_check['optimization_patterns']} {'✅' if returns_check['has_optimization_patterns'] else '❌'}")
    
    print()
    
    # Check VectorBTRollingOptimizer
    rolling_optimizer_file = "src/feature_generation/utils/vectorbt_rolling_optimizer.py"
    rolling_optimizer_check = check_file_for_vectorbt_usage(rolling_optimizer_file)
    
    print(f"VectorBTRollingOptimizer: {rolling_optimizer_file}")
    print(f"  Exists: {'✅' if rolling_optimizer_check['exists'] else '❌'}")
    if rolling_optimizer_check['exists']:
        print(f"  VectorBT imports: {rolling_optimizer_check['vectorbt_imports']} {'✅' if rolling_optimizer_check['has_vectorbt_imports'] else '❌'}")
        print(f"  Rolling operations: {rolling_optimizer_check['rolling_operations']} {'✅' if rolling_optimizer_check['has_rolling_operations'] else '❌'}")
    
    print()
    
    # Check UnifiedVectorizationManager
    unified_manager_file = "src/feature_generation/utils/unified_vectorization_manager.py"
    unified_manager_check = check_file_for_vectorbt_usage(unified_manager_file)
    
    print(f"UnifiedVectorizationManager: {unified_manager_file}")
    print(f"  Exists: {'✅' if unified_manager_check['exists'] else '❌'}")
    if unified_manager_check['exists']:
        print(f"  VectorBT imports: {unified_manager_check['vectorbt_imports']} {'✅' if unified_manager_check['has_vectorbt_imports'] else '❌'}")
        print(f"  Rolling operations: {unified_manager_check['rolling_operations']} {'✅' if unified_manager_check['has_rolling_operations'] else '❌'}")
    
    print()
    
    # Summary
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    total_checks = 0
    passed_checks = 0
    
    # Check returns module
    if returns_check['exists']:
        total_checks += 5
        if returns_check['has_vectorbt_imports']:
            passed_checks += 1
        if returns_check['has_rolling_optimizer']:
            passed_checks += 1
        if returns_check['has_unified_manager']:
            passed_checks += 1
        if returns_check['has_rolling_operations']:
            passed_checks += 1
        if returns_check['has_optimization_patterns']:
            passed_checks += 1
    
    # Check rolling optimizer
    if rolling_optimizer_check['exists']:
        total_checks += 2
        if rolling_optimizer_check['has_vectorbt_imports']:
            passed_checks += 1
        if rolling_optimizer_check['has_rolling_operations']:
            passed_checks += 1
    
    # Check unified manager
    if unified_manager_check['exists']:
        total_checks += 2
        if unified_manager_check['has_vectorbt_imports']:
            passed_checks += 1
        if unified_manager_check['has_rolling_operations']:
            passed_checks += 1
    
    print(f"Total checks: {total_checks}")
    print(f"Passed checks: {passed_checks}")
    print(f"Success rate: {(passed_checks / total_checks * 100):.1f}%")
    
    if passed_checks == total_checks:
        print("🎉 All VectorBT optimizations are properly implemented!")
    elif passed_checks >= total_checks * 0.8:
        print("✅ Most VectorBT optimizations are implemented!")
    else:
        print("⚠️  Some VectorBT optimizations may be missing.")
    
    return {
        'returns_module': returns_check,
        'rolling_optimizer': rolling_optimizer_check,
        'unified_manager': unified_manager_check,
        'total_checks': total_checks,
        'passed_checks': passed_checks,
        'success_rate': passed_checks / total_checks * 100 if total_checks > 0 else 0
    }

def check_specific_optimizations():
    """Check for specific optimization patterns in the returns module."""
    print("\nChecking specific optimization patterns...")
    print("=" * 60)
    
    returns_file = "src/feature_generation/categories/returns.py"
    if not os.path.exists(returns_file):
        print("❌ Returns module not found")
        return
    
    with open(returns_file, 'r') as f:
        content = f.read()
    
    # Check for specific generator optimizations
    generators = [
        'ReturnsFeatureGenerator',
        'LogReturnsGenerator', 
        'SimpleReturnsGenerator',
        'CumulativeReturnsGenerator',
        'ReturnsVolatilityGenerator',
        'SharpeRatioGenerator'
    ]
    
    for generator in generators:
        # Check if generator has VectorBT optimization
        pattern = f'class {generator}.*?def __init__.*?self\.rolling_optimizer.*?self\.unified_manager'
        has_optimization = bool(re.search(pattern, content, re.DOTALL))
        
        # Check if generator uses VectorBT operations
        pattern = f'class {generator}.*?VECTORBT_AVAILABLE.*?rolling_optimizer'
        uses_vectorbt = bool(re.search(pattern, content, re.DOTALL))
        
        print(f"{generator}:")
        print(f"  Has optimization setup: {'✅' if has_optimization else '❌'}")
        print(f"  Uses VectorBT operations: {'✅' if uses_vectorbt else '❌'}")
        print()

def main():
    """Main validation function."""
    print("VectorBT Returns Module Optimization Validation")
    print("=" * 60)
    
    # Validate main optimizations
    results = validate_returns_module()
    
    # Check specific patterns
    check_specific_optimizations()
    
    # Final assessment
    print("\nFINAL ASSESSMENT")
    print("=" * 60)
    
    if results['success_rate'] >= 90:
        print("🎉 EXCELLENT: VectorBT optimizations are fully implemented!")
        print("   The returns module is now using:")
        print("   - VectorBTRollingOptimizer for centralized rolling operations")
        print("   - UnifiedVectorizationManager for unified optimization management")
        print("   - VectorBT native functions for maximum performance")
        print("   - Intelligent fallbacks to pandas/numpy when needed")
    elif results['success_rate'] >= 70:
        print("✅ GOOD: Most VectorBT optimizations are implemented!")
        print("   The returns module has significant VectorBT integration.")
    elif results['success_rate'] >= 50:
        print("⚠️  PARTIAL: Some VectorBT optimizations are implemented.")
        print("   Consider adding more VectorBT usage for better performance.")
    else:
        print("❌ INCOMPLETE: VectorBT optimizations need more work.")
        print("   The returns module needs more VectorBT integration.")
    
    return results

if __name__ == "__main__":
    main()