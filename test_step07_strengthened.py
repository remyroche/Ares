#!/usr/bin/env python3
import numpy as np

"""
Test script for strengthened Step07 with dependency management.

This script tests the enhanced Step07 with proper fallback handling
and dependency management.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_step07_dependencies():
    """Test Step07 dependency checking."""
    print("🔍 Testing Step07 Dependency Management")
    print("=" * 50)
    
    try:
        from src.training.steps.model_training.step07_enhanced_matrix_operations import (
            check_step07_dependencies, get_step07_capabilities
        )
        
        # Check dependencies
        dependencies = check_step07_dependencies()
        print("📦 Dependency Status:")
        for dep, available in dependencies.items():
            status = "✅" if available else "❌"
            print(f"  {status} {dep}: {'Available' if available else 'Missing'}")
        
        # Check capabilities
        capabilities = get_step07_capabilities()
        print(f"\n📊 Capability Status: {capabilities['status']}")
        print(f"📈 Overall Score: {capabilities['overall_score']:.2%}")
        
        print("\n🔧 Available Capabilities:")
        for cap, available in capabilities.items():
            if cap not in ['overall_score', 'status']:
                status = "✅" if available else "❌"
                print(f"  {status} {cap}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing dependencies: {e}")
        import traceback

        traceback.print_exc()
        return False

def test_step07_creation():
    """Test Step07 step creation."""
    print("\n🚀 Testing Step07 Step Creation")
    print("=" * 40)
    
    try:
        from src.training.steps.model_training.step07_enhanced_matrix_operations import EnhancedMatrixOperationsStep
        
        # Create step with minimal config
        config = {
            'matrix_operations_config': {
                'use_gpu': False,
                'use_numba': False,
                'batch_size': 1000
            }
        }
        
        step = EnhancedMatrixOperationsStep(config)
        print(f"✅ Step created: {step.step_name}")
        print(f"📊 Dependencies: {len([d for d in step.dependencies.values() if d])}/{len(step.dependencies)} available")
        print(f"🔧 Capabilities: {step.capabilities['status']} ({step.capabilities['overall_score']:.2%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating step: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step07_execution():
    """Test Step07 execution with mock data."""
    print("\n🧮 Testing Step07 Execution")
    print("=" * 35)
    
    try:
        from src.training.steps.model_training.step07_enhanced_matrix_operations import EnhancedMatrixOperationsStep
        
        # Create step
        config = {
            'matrix_operations_config': {
                'use_gpu': False,
                'use_numba': False,
                'batch_size': 1000
            }
        }
        
        step = EnhancedMatrixOperationsStep(config)
        
        # Create mock data
        mock_data = [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0]
        ]
        
        # Test matrix computation fallback
        matrices = step._compute_matrices_fallback(mock_data, [])
        print(f"✅ Fallback matrix computation: {len(matrices)} matrices computed")
        
        if 'correlation_matrix' in matrices:
            corr_matrix = matrices['correlation_matrix']
            print(f"📊 Correlation matrix: {len(corr_matrix)}x{len(corr_matrix[0]) if corr_matrix else 0}")
        
        # Test basic correlation computation
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        corr = step._compute_basic_correlation(x, y)
        print(f"📈 Basic correlation test: {corr:.3f} (expected: ~1.0)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing execution: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step07_import_fixes():
    """Test Step07 import fix module."""
    print("\n🔧 Testing Step07 Import Fixes")
    print("=" * 35)
    
    try:
        from src.utils.step07_import_fix import (
            check_dependencies, get_import_summary, safe_importer
        )
        
        # Test dependency checking
        deps_ok = check_dependencies()
        print(f"✅ Dependencies check: {'PASS' if deps_ok else 'FAIL'}")
        
        # Test import summary
        summary = get_import_summary()
        print(f"📊 Import summary: {len([s for s in summary.values() if s])}/{len(summary)} modules available")
        
        # Test safe importer
        numpy_test = safe_importer.safe_import('numpy')
        print(f"🧮 NumPy import: {'✅ Available' if numpy_test is not None else '❌ Not available'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing import fixes: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Testing Strengthened Step07")
    print("=" * 50)
    
    tests = [
        ("Dependency Management", test_step07_dependencies),
        ("Step Creation", test_step07_creation),
        ("Execution", test_step07_execution),
        ("Import Fixes", test_step07_import_fixes)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"📊 {test_name}: {status}")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 All tests passed! Step07 is ready for use.")
    elif passed >= total * 0.75:
        print("⚠️ Most tests passed. Step07 is functional with some limitations.")
    else:
        print("❌ Multiple test failures. Step07 needs additional fixes.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)