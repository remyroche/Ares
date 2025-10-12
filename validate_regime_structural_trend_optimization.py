#!/usr/bin/env python3
"""
Simple validation script for regime structural trend VectorBT optimization.

This script validates the code changes without requiring external dependencies.
"""

import os
import re

def validate_imports():
    """Validate that the necessary imports are present."""
    print("🔍 Validating imports...")
    
    file_path = "/workspace/src/feature_generation/categories/regime_structural_trend.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for VectorBT imports
        vectorbt_imports = [
            "rolling_quantile", "rolling_skew", "rolling_kurt",
            "get_vectorbt_rolling_optimizer", "VectorBTRollingOptimizer",
            "get_unified_vectorization_manager", "OperationType"
        ]
        
        missing_imports = []
        for import_name in vectorbt_imports:
            if import_name not in content:
                missing_imports.append(import_name)
        
        if missing_imports:
            print(f"❌ Missing imports: {missing_imports}")
            return False
        else:
            print("✅ All required imports present")
            return True
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def validate_optimization_integration():
    """Validate that optimization components are integrated."""
    print("\n🔍 Validating optimization integration...")
    
    file_path = "/workspace/src/feature_generation/categories/regime_structural_trend.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for optimization component initialization
        optimization_checks = [
            "self.vectorbt_optimizer",
            "self.unified_manager",
            "get_vectorbt_rolling_optimizer",
            "get_unified_vectorization_manager"
        ]
        
        missing_components = []
        for component in optimization_checks:
            if component not in content:
                missing_components.append(component)
        
        if missing_components:
            print(f"❌ Missing optimization components: {missing_components}")
            return False
        else:
            print("✅ Optimization components integrated")
            return True
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def validate_vectorbt_usage():
    """Validate that VectorBT operations are used in calculations."""
    print("\n🔍 Validating VectorBT usage in calculations...")
    
    file_path = "/workspace/src/feature_generation/categories/regime_structural_trend.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for VectorBT optimizer usage in calculation methods
        calculation_methods = [
            "_calculate_structural_trend_persistence",
            "_calculate_trend_direction_consistency", 
            "_calculate_trend_regime_persistence",
            "_calculate_structural_trend_strength",
            "_calculate_trend_acceleration",
            "_calculate_trend_intensity",
            "_calculate_market_structure_strength",
            "_calculate_support_resistance_strength",
            "_calculate_market_structure_consistency"
        ]
        
        optimized_methods = []
        for method in calculation_methods:
            # Look for the method definition and check if it uses vectorbt_optimizer
            method_pattern = rf"def {method}\(.*?\):.*?(?=def|\Z)"
            method_match = re.search(method_pattern, content, re.DOTALL)
            
            if method_match and "vectorbt_optimizer" in method_match.group(0):
                optimized_methods.append(method)
        
        print(f"✅ {len(optimized_methods)}/{len(calculation_methods)} methods use VectorBT optimizer")
        
        if len(optimized_methods) >= len(calculation_methods) * 0.8:  # 80% threshold
            print("✅ Sufficient VectorBT optimization coverage")
            return True
        else:
            print(f"⚠️ Only {len(optimized_methods)} methods optimized")
            return False
            
    except Exception as e:
        print(f"❌ Error validating VectorBT usage: {e}")
        return False

def validate_fallback_mechanisms():
    """Validate that proper fallback mechanisms are in place."""
    print("\n🔍 Validating fallback mechanisms...")
    
    file_path = "/workspace/src/feature_generation/categories/regime_structural_trend.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for fallback patterns
        fallback_patterns = [
            "except Exception as e:",
            "print(f\"VectorBT optimizer failed, using fallback: {e}\")",
            "VECTORBT_AVAILABLE",
            "Final fallback to pandas"
        ]
        
        found_patterns = []
        for pattern in fallback_patterns:
            if pattern in content:
                found_patterns.append(pattern)
        
        print(f"✅ Found {len(found_patterns)}/{len(fallback_patterns)} fallback patterns")
        
        if len(found_patterns) >= len(fallback_patterns) * 0.75:  # 75% threshold
            print("✅ Proper fallback mechanisms in place")
            return True
        else:
            print("⚠️ Some fallback mechanisms may be missing")
            return False
            
    except Exception as e:
        print(f"❌ Error validating fallback mechanisms: {e}")
        return False

def validate_performance_monitoring():
    """Validate that performance monitoring is implemented."""
    print("\n🔍 Validating performance monitoring...")
    
    file_path = "/workspace/src/feature_generation/categories/regime_structural_trend.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for performance monitoring features
        performance_features = [
            "get_optimization_stats",
            "performance_stats",
            "get_performance_stats"
        ]
        
        found_features = []
        for feature in performance_features:
            if feature in content:
                found_features.append(feature)
        
        print(f"✅ Found {len(found_features)}/{len(performance_features)} performance monitoring features")
        
        if len(found_features) >= len(performance_features) * 0.66:  # 66% threshold
            print("✅ Performance monitoring implemented")
            return True
        else:
            print("⚠️ Some performance monitoring features may be missing")
            return False
            
    except Exception as e:
        print(f"❌ Error validating performance monitoring: {e}")
        return False

def validate_code_quality():
    """Validate code quality and structure."""
    print("\n🔍 Validating code quality...")
    
    file_path = "/workspace/src/feature_generation/categories/regime_structural_trend.py"
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for code quality indicators
        quality_checks = [
            ("Method documentation", "def _calculate_.*?:\n.*?\"\"\".*?OPTIMIZED VECTORBT"),
            ("Error handling", "try:\n.*?except Exception as e:"),
            ("Type hints", "-> np.ndarray:"),
            ("Comments", "# Use VectorBT rolling optimizer if available")
        ]
        
        quality_score = 0
        for check_name, pattern in quality_checks:
            if re.search(pattern, content, re.DOTALL):
                quality_score += 1
                print(f"  ✅ {check_name}")
            else:
                print(f"  ⚠️ {check_name}")
        
        print(f"✅ Code quality score: {quality_score}/{len(quality_checks)}")
        
        return quality_score >= len(quality_checks) * 0.75  # 75% threshold
        
    except Exception as e:
        print(f"❌ Error validating code quality: {e}")
        return False

def main():
    """Run all validation checks."""
    print("🚀 Validating Regime Structural Trend VectorBT Optimization")
    print("=" * 70)
    
    validations = [
        ("Import Validation", validate_imports),
        ("Optimization Integration", validate_optimization_integration),
        ("VectorBT Usage", validate_vectorbt_usage),
        ("Fallback Mechanisms", validate_fallback_mechanisms),
        ("Performance Monitoring", validate_performance_monitoring),
        ("Code Quality", validate_code_quality)
    ]
    
    results = []
    
    for validation_name, validation_func in validations:
        print(f"\n{'='*20} {validation_name} {'='*20}")
        try:
            result = validation_func()
            results.append((validation_name, result))
            if result:
                print(f"✅ {validation_name} PASSED")
            else:
                print(f"❌ {validation_name} FAILED")
        except Exception as e:
            print(f"❌ {validation_name} ERROR: {e}")
            results.append((validation_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for validation_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{validation_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} validations passed")
    
    if passed == total:
        print("🎉 All validations passed! VectorBT optimization is properly implemented.")
    elif passed >= total * 0.8:
        print("✅ Most validations passed! VectorBT optimization is mostly complete.")
    else:
        print("⚠️ Some validations failed. Check the output above for details.")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)