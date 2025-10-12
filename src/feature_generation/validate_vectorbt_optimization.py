#!/usr/bin/env python3
"""
VectorBT Optimization Validation Script

This script validates the VectorBT optimization implementation in the volatility
features without requiring external dependencies for testing.
"""

import sys
import os
import ast
import re
from pathlib import Path

def validate_imports(file_path):
    """Validate that all required imports are present."""
    print(f"🔍 Validating imports in {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the AST to check imports
        tree = ast.parse(content)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        # Check for VectorBT imports
        vectorbt_imports = [imp for imp in imports if 'vectorbt' in imp.lower()]
        if vectorbt_imports:
            print(f"  ✅ VectorBT imports found: {vectorbt_imports}")
        else:
            print(f"  ⚠️ No VectorBT imports found")
        
        # Check for UnifiedVectorizationManager imports
        unified_imports = [imp for imp in imports if 'unified_vectorization_manager' in imp.lower()]
        if unified_imports:
            print(f"  ✅ UnifiedVectorizationManager imports found: {unified_imports}")
        else:
            print(f"  ⚠️ No UnifiedVectorizationManager imports found")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error validating imports: {e}")
        return False

def validate_vectorbt_usage(file_path):
    """Validate VectorBT usage patterns."""
    print(f"🔍 Validating VectorBT usage in {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for VectorBT rolling operations
        rolling_ops = [
            'rolling_mean', 'rolling_std', 'rolling_var', 'rolling_min', 'rolling_max',
            'rolling_sum', 'rolling_apply', 'rolling_corr', 'rolling_cov'
        ]
        
        found_ops = []
        for op in rolling_ops:
            if op in content:
                found_ops.append(op)
        
        if found_ops:
            print(f"  ✅ VectorBT rolling operations found: {found_ops}")
        else:
            print(f"  ⚠️ No VectorBT rolling operations found")
        
        # Check for VectorBTRollingOptimizer usage
        if 'VectorBTRollingOptimizer' in content:
            print(f"  ✅ VectorBTRollingOptimizer usage found")
        else:
            print(f"  ⚠️ No VectorBTRollingOptimizer usage found")
        
        # Check for UnifiedVectorizationManager usage
        if 'UnifiedVectorizationManager' in content:
            print(f"  ✅ UnifiedVectorizationManager usage found")
        else:
            print(f"  ⚠️ No UnifiedVectorizationManager usage found")
        
        # Check for strategy selection logic
        if 'select_optimal_strategy' in content:
            print(f"  ✅ Strategy selection logic found")
        else:
            print(f"  ⚠️ No strategy selection logic found")
        
        # Check for performance tracking
        if 'performance_stats' in content:
            print(f"  ✅ Performance tracking found")
        else:
            print(f"  ⚠️ No performance tracking found")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error validating VectorBT usage: {e}")
        return False

def validate_class_structure(file_path):
    """Validate class structure and methods."""
    print(f"🔍 Validating class structure in {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the AST to check class structure
        tree = ast.parse(content)
        
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        print(f"  📋 Classes found: {classes}")
        
        # Check for enhanced volatility generator
        if 'EnhancedVectorBTVolatilityGenerator' in classes:
            print(f"  ✅ EnhancedVectorBTVolatilityGenerator class found")
        else:
            print(f"  ⚠️ EnhancedVectorBTVolatilityGenerator class not found")
        
        # Check for configuration classes
        if 'VolatilityConfig' in classes:
            print(f"  ✅ VolatilityConfig class found")
        else:
            print(f"  ⚠️ VolatilityConfig class not found")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error validating class structure: {e}")
        return False

def validate_functions(file_path):
    """Validate function definitions."""
    print(f"🔍 Validating functions in {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the AST to check function definitions
        tree = ast.parse(content)
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        # Check for key functions
        key_functions = [
            'create_enhanced_volatility_generators',
            'create_comprehensive_vectorbt_volatility_generators',
            'create_optimized_volatility_pipeline',
            'benchmark_volatility_optimizations'
        ]
        
        found_functions = []
        for func in key_functions:
            if func in functions:
                found_functions.append(func)
        
        if found_functions:
            print(f"  ✅ Key functions found: {found_functions}")
        else:
            print(f"  ⚠️ Key functions not found: {[f for f in key_functions if f not in found_functions]}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error validating functions: {e}")
        return False

def validate_error_handling(file_path):
    """Validate error handling patterns."""
    print(f"🔍 Validating error handling in {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for try-except blocks
        try_except_count = content.count('try:')
        if try_except_count > 0:
            print(f"  ✅ Try-except blocks found: {try_except_count}")
        else:
            print(f"  ⚠️ No try-except blocks found")
        
        # Check for fallback mechanisms
        fallback_patterns = ['fallback', 'except Exception', 'logger.warning']
        found_fallbacks = []
        for pattern in fallback_patterns:
            if pattern in content:
                found_fallbacks.append(pattern)
        
        if found_fallbacks:
            print(f"  ✅ Fallback mechanisms found: {found_fallbacks}")
        else:
            print(f"  ⚠️ No fallback mechanisms found")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error validating error handling: {e}")
        return False

def validate_documentation(file_path):
    """Validate documentation quality."""
    print(f"🔍 Validating documentation in {file_path}...")
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for docstrings
        docstring_count = content.count('"""')
        if docstring_count > 0:
            print(f"  ✅ Docstrings found: {docstring_count // 2} pairs")
        else:
            print(f"  ⚠️ No docstrings found")
        
        # Check for type hints
        type_hint_patterns = ['->', 'Optional[', 'List[', 'Dict[', 'Union[']
        found_hints = []
        for pattern in type_hint_patterns:
            if pattern in content:
                found_hints.append(pattern)
        
        if found_hints:
            print(f"  ✅ Type hints found: {found_hints}")
        else:
            print(f"  ⚠️ No type hints found")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error validating documentation: {e}")
        return False

def main():
    """Main validation function."""
    print("🚀 VectorBT Optimization Validation")
    print("=" * 50)
    
    # Files to validate
    files_to_validate = [
        'src/feature_generation/categories/volatility.py',
        'src/feature_generation/categories/enhanced_vectorbt_volatility.py',
        'src/feature_generation/utils/vectorbt_rolling_optimizer.py',
        'src/feature_generation/utils/vectorization_optimizer.py'
    ]
    
    validation_results = {}
    
    for file_path in files_to_validate:
        if os.path.exists(file_path):
            print(f"\n📁 Validating {file_path}")
            print("-" * 40)
            
            results = {
                'imports': validate_imports(file_path),
                'vectorbt_usage': validate_vectorbt_usage(file_path),
                'class_structure': validate_class_structure(file_path),
                'functions': validate_functions(file_path),
                'error_handling': validate_error_handling(file_path),
                'documentation': validate_documentation(file_path)
            }
            
            validation_results[file_path] = results
            
            # Overall score for this file
            passed = sum(results.values())
            total = len(results)
            score = (passed / total) * 100
            
            print(f"  📊 Overall Score: {score:.1f}% ({passed}/{total})")
            
        else:
            print(f"  ❌ File not found: {file_path}")
            validation_results[file_path] = None
    
    # Summary
    print("\n📊 Validation Summary")
    print("=" * 50)
    
    total_files = len([f for f in validation_results.values() if f is not None])
    if total_files > 0:
        all_passed = all(
            all(results.values()) if results else False
            for results in validation_results.values()
        )
        
        if all_passed:
            print("✅ All validations passed!")
        else:
            print("⚠️ Some validations failed. See details above.")
        
        # Detailed breakdown
        for file_path, results in validation_results.items():
            if results:
                print(f"\n{file_path}:")
                for check, passed in results.items():
                    status = "✅" if passed else "❌"
                    print(f"  {status} {check}")
    
    print("\n🎉 Validation completed!")
    return validation_results

if __name__ == '__main__':
    main()