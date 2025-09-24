#!/usr/bin/env python3
"""
Basic TAS Regime Integration Test

This script tests basic integration without external dependencies.
"""

import sys
import os

# Add the project root to the path
sys.path.append('/workspace/src')

def test_directory_structure():
    """Test that the tas_regime directory structure is correct."""
    print("🔍 Testing directory structure...")
    
    required_dirs = [
        'core',
        'components', 
        'evaluation',
        'meta_learning',
        'optimization',
        'regime_analysis',
        'search',
        'adaptation',
        'uncertainty',
        'utils',
        'examples'
    ]
    
    base_path = '/workspace/src/training/steps/market_analysis/tas_regime'
    
    for dir_name in required_dirs:
        dir_path = os.path.join(base_path, dir_name)
        if os.path.exists(dir_path):
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            return False
    
    return True

def test_core_files():
    """Test that core files exist."""
    print("\n🔍 Testing core files...")
    
    core_files = [
        'core/tas_config.py',
        'core/tas_engine.py', 
        'core/tas_result.py',
        'core/tree_architecture.py',
        'core/tree_cvlSA_architecture.py',
        'core/advanced_tas_search.py',
        '__init__.py',
        'README.md'
    ]
    
    base_path = '/workspace/src/training/steps/market_analysis/tas_regime'
    
    for file_name in core_files:
        file_path = os.path.join(base_path, file_name)
        if os.path.exists(file_path):
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
            return False
    
    return True

def test_import_structure():
    """Test that import structure is correct."""
    print("\n🔍 Testing import structure...")
    
    try:
        # Test that we can import the main module
        sys.path.insert(0, '/workspace/src')
        from src.training.steps.market_analysis.tas_regime import __init__
        print("✅ Main __init__.py imports successfully")
        
        # Test that we can access the module
        import src.training.steps.market_analysis.tas_regime as tas_regime
        print("✅ tas_regime module accessible")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_file_content():
    """Test that key files have expected content."""
    print("\n🔍 Testing file content...")
    
    # Test README.md
    readme_path = '/workspace/src/training/steps/market_analysis/tas_regime/README.md'
    if os.path.exists(readme_path):
        with open(readme_path, 'r') as f:
            content = f.read()
            if 'TAS' in content and 'Tree Architecture Search' in content:
                print("✅ README.md has expected content")
            else:
                print("❌ README.md missing expected content")
                return False
    else:
        print("❌ README.md not found")
        return False
    
    # Test __init__.py
    init_path = '/workspace/src/training/steps/market_analysis/tas_regime/__init__.py'
    if os.path.exists(init_path):
        with open(init_path, 'r') as f:
            content = f.read()
            if 'TreeArchitectureSearchEngine' in content:
                print("✅ __init__.py has expected exports")
            else:
                print("❌ __init__.py missing expected exports")
                return False
    else:
        print("❌ __init__.py not found")
        return False
    
    return True

def main():
    """Run all basic integration tests."""
    print("🧪 Testing TAS Regime Basic Integration")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Core Files", test_core_files),
        ("Import Structure", test_import_structure),
        ("File Content", test_file_content)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TAS regime integration is working.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)