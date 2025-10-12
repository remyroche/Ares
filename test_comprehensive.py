#!/usr/bin/env python3
"""
Comprehensive test script to verify duplicate cleanup was successful.
This script tests syntax, imports, and basic functionality.
"""

import sys
import os
import ast
import traceback

def test_syntax(file_path):
    """Test if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except Exception as e:
        return False, str(e)

def test_imports():
    """Test if core modules can be imported."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        # Test core imports
        from feature_generation.core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
        print("✅ Core imports successful")
        
        # Test that base class methods exist
        config = FeatureConfig(
            name="test",
            category=FeatureCategory.CUSTOM,
            description="Test",
            required_columns=["close"]
        )
        
        generator = VectorizedFeatureGenerator(config)
        
        # Test that methods exist
        assert hasattr(generator, 'optimize_dataframe_processing'), "optimize_dataframe_processing method missing"
        assert hasattr(generator, 'vectorized_rolling_operations'), "vectorized_rolling_operations method missing"
        print("✅ Base class methods exist")
        
        return True, None
        
    except Exception as e:
        return False, str(e)

def test_duplicate_removal():
    """Test that duplicates were actually removed."""
    try:
        import subprocess
        
        # Count remaining duplicates
        result = subprocess.run([
            'grep', '-r', 'def optimize_dataframe_processing', 'src/feature_generation/categories/'
        ], capture_output=True, text=True)
        
        duplicate_count = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        
        # Should only have a few specialized implementations left
        if duplicate_count <= 5:  # Allow for some specialized implementations
            print(f"✅ Duplicate removal successful: {duplicate_count} remaining (expected ≤5)")
            return True, None
        else:
            return False, f"Too many duplicates remaining: {duplicate_count}"
            
    except Exception as e:
        return False, str(e)

def main():
    """Run comprehensive tests."""
    print("🧪 Running comprehensive duplicate cleanup tests...\n")
    
    # Files to test
    files_to_test = [
        'src/feature_generation/core/feature_generator.py',
        'src/feature_generation/core/vectorbt_optimization_mixin.py',
        'src/feature_generation/categories/volume.py',
        'src/feature_generation/categories/acceleration.py',
        'src/feature_generation/categories/entropy.py',
        'src/feature_generation/categories/oscillator.py',
        'src/feature_generation/categories/interaction.py',
        'src/feature_generation/categories/cross_timeframe.py',
        'src/feature_generation/categories/autoencoder.py',
        'src/feature_generation/categories/representation_learning.py',
        'src/feature_generation/categories/regime_feature_integration.py'
    ]
    
    # Test 1: Syntax validation
    print("📝 Testing syntax validation...")
    syntax_passed = 0
    syntax_failed = []
    
    for file_path in files_to_test:
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue
            
        syntax_ok, error = test_syntax(file_path)
        if syntax_ok:
            print(f"  ✅ {file_path}")
            syntax_passed += 1
        else:
            print(f"  ❌ {file_path}: {error}")
            syntax_failed.append((file_path, error))
    
    print(f"📊 Syntax Results: {syntax_passed}/{len(files_to_test)} files passed\n")
    
    # Test 2: Import validation
    print("📦 Testing import validation...")
    import_ok, import_error = test_imports()
    if import_ok:
        print("✅ Import validation passed\n")
    else:
        print(f"❌ Import validation failed: {import_error}\n")
    
    # Test 3: Duplicate removal validation
    print("🔄 Testing duplicate removal...")
    duplicate_ok, duplicate_error = test_duplicate_removal()
    if duplicate_ok:
        print("✅ Duplicate removal validation passed\n")
    else:
        print(f"❌ Duplicate removal validation failed: {duplicate_error}\n")
    
    # Summary
    total_tests = 3
    passed_tests = sum([
        syntax_passed == len(files_to_test),
        import_ok,
        duplicate_ok
    ])
    
    print(f"📊 Overall Results: {passed_tests}/{total_tests} test categories passed")
    
    if syntax_failed:
        print("\n❌ Syntax errors found:")
        for file_path, error in syntax_failed:
            print(f"  - {file_path}: {error}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Duplicate cleanup was successful.")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} test categories failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)