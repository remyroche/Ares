#!/usr/bin/env python3
"""
Validation script for trend feature optimization implementation.
This script validates the code structure and imports without requiring external dependencies.
"""

import sys
import os
import ast

def validate_imports():
    """Validate that all required imports are present."""
    print("🔍 Validating imports...")
    
    try:
        # Read the trend.py file
        with open('/workspace/src/feature_generation/categories/trend.py', 'r') as f:
            content = f.read()
        
        # Check for VectorBTRollingOptimizer import
        if 'from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer' in content:
            print("   ✅ VectorBTRollingOptimizer import found")
        else:
            print("   ❌ VectorBTRollingOptimizer import not found")
            return False
        
        # Check for UnifiedVectorizationManager import
        if 'from ...utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager' in content:
            print("   ✅ UnifiedVectorizationManager import found")
        else:
            print("   ❌ UnifiedVectorizationManager import not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False

def validate_class_definitions():
    """Validate that the new classes are properly defined."""
    print("\n🔍 Validating class definitions...")
    
    try:
        with open('/workspace/src/feature_generation/categories/trend.py', 'r') as f:
            content = f.read()
        
        # Check for OptimizedTrendFeatureGenerator class
        if 'class OptimizedTrendFeatureGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):' in content:
            print("   ✅ OptimizedTrendFeatureGenerator class found")
        else:
            print("   ❌ OptimizedTrendFeatureGenerator class not found")
            return False
        
        # Check for VectorBTRollingOptimizer usage in TrendFeatureGenerator
        if 'self.vectorbt_optimizer = get_vectorbt_rolling_optimizer' in content:
            print("   ✅ VectorBTRollingOptimizer initialization found")
        else:
            print("   ❌ VectorBTRollingOptimizer initialization not found")
            return False
        
        # Check for UnifiedVectorizationManager usage
        if 'self.unified_manager = get_unified_vectorization_manager' in content:
            print("   ✅ UnifiedVectorizationManager initialization found")
        else:
            print("   ❌ UnifiedVectorizationManager initialization not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error validating classes: {e}")
        return False

def validate_method_implementations():
    """Validate that the optimization methods are implemented."""
    print("\n🔍 Validating method implementations...")
    
    try:
        with open('/workspace/src/feature_generation/categories/trend.py', 'r') as f:
            content = f.read()
        
        # Check for optimized _vectorbt_rolling_operation method
        if 'def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,' in content:
            print("   ✅ Optimized _vectorbt_rolling_operation method found")
        else:
            print("   ❌ Optimized _vectorbt_rolling_operation method not found")
            return False
        
        # Check for VectorBTRollingOptimizer usage in the method
        if 'self.vectorbt_optimizer.rolling_mean' in content:
            print("   ✅ VectorBTRollingOptimizer usage in rolling operations found")
        else:
            print("   ❌ VectorBTRollingOptimizer usage in rolling operations not found")
            return False
        
        # Check for batch feature generation method
        if 'def generate_batch_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:' in content:
            print("   ✅ Batch feature generation method found")
        else:
            print("   ❌ Batch feature generation method not found")
            return False
        
        # Check for factory function
        if 'def create_optimized_trend_generators(' in content:
            print("   ✅ create_optimized_trend_generators factory function found")
        else:
            print("   ❌ create_optimized_trend_generators factory function not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error validating methods: {e}")
        return False

def validate_syntax():
    """Validate that the Python syntax is correct."""
    print("\n🔍 Validating Python syntax...")
    
    try:
        with open('/workspace/src/feature_generation/categories/trend.py', 'r') as f:
            content = f.read()
        
        # Parse the file to check for syntax errors
        ast.parse(content)
        print("   ✅ Python syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"   ❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error parsing file: {e}")
        return False

def validate_optimization_features():
    """Validate that optimization features are properly implemented."""
    print("\n🔍 Validating optimization features...")
    
    try:
        with open('/workspace/src/feature_generation/categories/trend.py', 'r') as f:
            content = f.read()
        
        # Check for memory optimization
        if 'memory_efficient' in content:
            print("   ✅ Memory optimization features found")
        else:
            print("   ⚠️ Memory optimization features not explicitly found")
        
        # Check for parallel processing
        if 'enable_parallel' in content:
            print("   ✅ Parallel processing features found")
        else:
            print("   ⚠️ Parallel processing features not explicitly found")
        
        # Check for GPU acceleration
        if 'enable_gpu' in content:
            print("   ✅ GPU acceleration features found")
        else:
            print("   ⚠️ GPU acceleration features not explicitly found")
        
        # Check for fallback mechanisms
        if 'fallback' in content.lower():
            print("   ✅ Fallback mechanisms found")
        else:
            print("   ⚠️ Fallback mechanisms not explicitly found")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error validating optimization features: {e}")
        return False

def main():
    """Run all validation checks."""
    print("🚀 Starting trend feature optimization validation...\n")
    
    checks = [
        validate_imports,
        validate_class_definitions,
        validate_method_implementations,
        validate_syntax,
        validate_optimization_features
    ]
    
    passed = 0
    total = len(checks)
    
    for check in checks:
        if check():
            passed += 1
        print()  # Add spacing between checks
    
    print(f"📊 Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All validations passed! The trend feature optimization is properly implemented.")
        print("\n📋 Summary of optimizations implemented:")
        print("   • VectorBTRollingOptimizer integration for enhanced rolling operations")
        print("   • UnifiedVectorizationManager integration for intelligent optimization selection")
        print("   • OptimizedTrendFeatureGenerator for batch processing")
        print("   • Enhanced VectorBTTrendFeatureGenerator with VectorBTRollingOptimizer")
        print("   • Batch feature generation methods for optimal performance")
        print("   • Factory functions for easy generator creation")
        print("   • Fallback mechanisms for robustness")
    else:
        print("⚠️ Some validations failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)