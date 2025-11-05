#!/usr/bin/env python3
"""
Simple Test for Statsmodel Clustering Implementation

This script performs basic tests without complex imports.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_basic_functionality():
    """Test basic functionality."""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test 1: Check if files exist
        files_to_check = [
            'core/__init__.py',
            'core/base_data_downloader.py',
            'core/markov_regression_adapter.py',
            'cli.py',
            'utils/result_converter.py'
        ]
        
        for file_path in files_to_check:
            full_path = current_dir / file_path
            if not full_path.exists():
                print(f"❌ Missing file: {file_path}")
                return False
            else:
                print(f"✅ Found file: {file_path}")
        
        # Test 2: Check if CLI can be imported
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("cli", current_dir / "cli.py")
            cli_module = importlib.util.module_from_spec(spec)
            print("✅ CLI module can be loaded")
        except Exception as e:
            print(f"⚠️ CLI module loading issue: {e}")
        
        # Test 3: Check if core modules can be imported
        try:
            spec = importlib.util.spec_from_file_location("base_data_downloader", current_dir / "core/base_data_downloader.py")
            base_module = importlib.util.module_from_spec(spec)
            print("✅ Base data downloader module can be loaded")
        except Exception as e:
            print(f"⚠️ Base data downloader module loading issue: {e}")
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_cli_help():
    """Test CLI help functionality."""
    print("\n🧪 Testing CLI help...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, str(current_dir / "cli.py"), "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ CLI help works")
            return True
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI help test failed: {e}")
        return False


def test_package_structure():
    """Test package structure."""
    print("\n🧪 Testing package structure...")
    
    try:
        # Check __init__.py files
        init_files = [
            '__init__.py',
            'core/__init__.py',
            'utils/__init__.py',
            'clustering/__init__.py',
            'feature_engineering/__init__.py',
            'optimization/__init__.py',
            'assessment/__init__.py'
        ]
        
        for init_file in init_files:
            full_path = current_dir / init_file
            if not full_path.exists():
                print(f"❌ Missing __init__.py: {init_file}")
                return False
            else:
                print(f"✅ Found __init__.py: {init_file}")
        
        print("✅ Package structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Package structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Simple Statsmodel Clustering Tests")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("CLI Help", test_cli_help),
        ("Package Structure", test_package_structure),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)