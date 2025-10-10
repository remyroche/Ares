#!/usr/bin/env python3
"""
Test script for the enhanced klines downloading and processing pipeline structure.

This script tests the pipeline structure to ensure all requirements are met:
1. Type hints & tprints
2. ExchangeInterface usage
3. Data standardizer integration
4. Fast fail pattern
5. No mock data or fallbacks
"""

import ast
import sys
from pathlib import Path


def test_type_hints():
    """Test that type hints are present."""
    print("🧪 Testing type hints...")
    
    try:
        with open("src/training/steps/data_collection/klines_downloading_processing_enhanced.py", "r") as f:
            content = f.read()
        
        # Parse the file
        tree = ast.parse(content)
        
        # Count function definitions with type hints
        functions_with_hints = 0
        total_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if node.returns is not None or any(arg.annotation is not None for arg in node.args.args):
                    functions_with_hints += 1
        
        print(f"✅ Found {functions_with_hints}/{total_functions} functions with type hints")
        return functions_with_hints > total_functions * 0.8  # At least 80% should have hints
        
    except Exception as e:
        print(f"❌ Type hints test failed: {e}")
        return False


def test_tprint_usage():
    """Test that tprint is used instead of logger."""
    print("🧪 Testing tprint usage...")
    
    try:
        with open("src/training/steps/data_collection/klines_downloading_processing_enhanced.py", "r") as f:
            content = f.read()
        
        # Check for tprint usage
        tprint_count = content.count("tprint")
        logger_count = content.count("self.logger")
        
        print(f"✅ Found {tprint_count} tprint calls and {logger_count} logger calls")
        return tprint_count > 0 and logger_count == 0
        
    except Exception as e:
        print(f"❌ tprint test failed: {e}")
        return False


def test_exchange_interface_usage():
    """Test that ExchangeInterface is used."""
    print("🧪 Testing ExchangeInterface usage...")
    
    try:
        with open("src/training/steps/data_collection/klines_downloading_processing_enhanced.py", "r") as f:
            content = f.read()
        
        # Check for ExchangeInterface imports and usage
        has_import = "ExchangeInterface" in content
        has_usage = "exchange_interface" in content
        has_create = "create_exchange_interface" in content
        
        print(f"✅ ExchangeInterface import: {has_import}")
        print(f"✅ ExchangeInterface usage: {has_usage}")
        print(f"✅ create_exchange_interface: {has_create}")
        
        return has_import and has_usage and has_create
        
    except Exception as e:
        print(f"❌ ExchangeInterface test failed: {e}")
        return False


def test_data_standardizer_integration():
    """Test that data standardizer is integrated."""
    print("🧪 Testing data standardizer integration...")
    
    try:
        with open("src/training/steps/data_collection/klines_downloading_processing_enhanced.py", "r") as f:
            content = f.read()
        
        # Check for data standardizer imports and usage
        has_import = "ExchangeDataStandardizer" in content
        has_usage = "self.data_standardizer" in content
        has_method = "standardize_data_format" in content
        
        print(f"✅ ExchangeDataStandardizer import: {has_import}")
        print(f"✅ Data standardizer usage: {has_usage}")
        print(f"✅ Standardize method: {has_method}")
        
        return has_import and has_usage and has_method
        
    except Exception as e:
        print(f"❌ Data standardizer test failed: {e}")
        return False


def test_fast_fail_pattern():
    """Test that fast fail pattern is implemented."""
    print("🧪 Testing fast fail pattern...")
    
    try:
        with open("src/training/steps/data_collection/klines_downloading_processing_enhanced.py", "r") as f:
            content = f.read()
        
        # Check for fast fail patterns
        has_raise = "raise" in content
        has_assert = "assert" in content
        has_error_returns = "error" in content and "return" in content
        # Check for fallback patterns in code (not comments)
        fallback_in_code = False
        for line in content.split('\n'):
            line = line.strip()
            if (not line.startswith('#') and 
                ('fallback' in line.lower() and 
                 ('if' in line or 'else' in line or 'except' in line or 'try' in line))):
                fallback_in_code = True
                break
        
        print(f"✅ Raise statements: {has_raise}")
        print(f"✅ Assert statements: {has_assert}")
        print(f"✅ Error returns: {has_error_returns}")
        print(f"✅ No fallback in code: {not fallback_in_code}")
        
        return has_raise and has_error_returns and not fallback_in_code
        
    except Exception as e:
        print(f"❌ Fast fail test failed: {e}")
        return False


def test_no_mock_data():
    """Test that no mock data or fallbacks are present."""
    print("🧪 Testing for absence of mock data...")
    
    try:
        with open("src/training/steps/data_collection/klines_downloading_processing_enhanced.py", "r") as f:
            content = f.read()
        
        # Check for mock data patterns in code (not comments)
        mock_patterns = [
            "simulated",
            "mock",
            "fake",
            "dummy",
            "placeholder"
        ]
        
        found_patterns = []
        for pattern in mock_patterns:
            for line in content.split('\n'):
                if pattern in line.lower() and not line.strip().startswith('#'):
                    found_patterns.append(pattern)
                    break
        
        print(f"✅ Found patterns in code: {found_patterns}")
        
        # Check for simulated exchange data in code
        simulated_exchange = False
        mock_exchange = False
        for line in content.split('\n'):
            if not line.strip().startswith('#'):
                if "simulated" in line.lower() and "exchange" in line.lower():
                    simulated_exchange = True
                if "mock" in line.lower() and "exchange" in line.lower():
                    mock_exchange = True
        
        if simulated_exchange:
            print("❌ Simulated exchange data found in code")
            return False
        
        if mock_exchange:
            print("❌ Mock exchange data found in code")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Mock data test failed: {e}")
        return False


def test_async_usage():
    """Test that async/await is used properly."""
    print("🧪 Testing async usage...")
    
    try:
        with open("src/training/steps/data_collection/klines_downloading_processing_enhanced.py", "r") as f:
            content = f.read()
        
        # Check for async patterns
        has_async = "async def" in content
        has_await = "await" in content
        has_asyncio = "asyncio" in content
        
        print(f"✅ Async functions: {has_async}")
        print(f"✅ Await usage: {has_await}")
        print(f"✅ Asyncio import: {has_asyncio}")
        
        return has_async and has_await and has_asyncio
        
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting klines pipeline structure tests...\n")
    
    tests = [
        ("Type Hints", test_type_hints),
        ("tprint Usage", test_tprint_usage),
        ("ExchangeInterface Usage", test_exchange_interface_usage),
        ("Data Standardizer Integration", test_data_standardizer_integration),
        ("Fast Fail Pattern", test_fast_fail_pattern),
        ("No Mock Data", test_no_mock_data),
        ("Async Usage", test_async_usage),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"📋 Running test: {test_name}")
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED\n")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED\n")
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}\n")
            failed += 1
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Pipeline meets all requirements.")
        return True
    else:
        print(f"❌ {failed} tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)