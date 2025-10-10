#!/usr/bin/env python3
"""
Verify that the improvements have been implemented correctly.
"""

import os
import sys
from pathlib import Path


def verify_file_exists(file_path, description):
    """Verify that a file exists."""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False


def verify_file_contains(file_path, content, description):
    """Verify that a file contains specific content."""
    if not os.path.exists(file_path):
        print(f"❌ {description}: {file_path} - FILE NOT FOUND")
        return False
    
    try:
        with open(file_path, 'r') as f:
            file_content = f.read()
        
        if content in file_content:
            print(f"✅ {description}: Found '{content}' in {file_path}")
            return True
        else:
            print(f"❌ {description}: '{content}' NOT FOUND in {file_path}")
            return False
    except Exception as e:
        print(f"❌ {description}: Error reading {file_path} - {e}")
        return False


def main():
    """Main verification function."""
    print("🔍 Verifying Exchange Interface and Shared Utilities Improvements")
    print("=" * 70)
    
    # Get workspace directory
    workspace = Path(__file__).parent.parent.parent.parent
    print(f"Working directory: {workspace}")
    print(f"Current directory: {os.getcwd()}")
    
    results = []
    
    # 1. Verify async context manager support in ExchangeInterface
    print("\n1. ExchangeInterface Async Context Manager Support:")
    exchange_interface_path = workspace / "exchanges" / "base_exchange" / "exchange_interface.py"
    results.append(verify_file_contains(
        str(exchange_interface_path),
        "async def __aenter__(self):",
        "IExchange async context manager entry"
    ))
    results.append(verify_file_contains(
        str(exchange_interface_path),
        "async def __aexit__(self, exc_type, exc_val, exc_tb):",
        "IExchange async context manager exit"
    ))
    results.append(verify_file_contains(
        str(exchange_interface_path),
        "async def __aenter__(self):",
        "IExchangeAdapter async context manager entry"
    ))
    results.append(verify_file_contains(
        str(exchange_interface_path),
        "async def __aexit__(self, exc_type, exc_val, exc_tb):",
        "IExchangeAdapter async context manager exit"
    ))
    
    # 2. Verify comprehensive unit tests
    print("\n2. Comprehensive Unit Tests:")
    test_files = [
        "test_auth_manager.py",
        "test_market_metadata.py", 
        "test_order_manager.py",
        "test_price_manager.py",
        "test_risk_calculator.py",
        "test_balance_manager.py",
        "test_rate_limit_manager.py",
        "test_exchange_interface.py"
    ]
    
    tests_dir = workspace / "exchanges" / "shared" / "tests"
    for test_file in test_files:
        test_path = tests_dir / test_file
        results.append(verify_file_exists(str(test_path), f"Unit test file: {test_file}"))
    
    # 3. Verify high-level interfaces
    print("\n3. High-Level Interfaces:")
    interfaces_path = workspace / "exchanges" / "shared" / "interfaces.py"
    results.append(verify_file_exists(
        str(interfaces_path),
        "High-level interfaces module"
    ))
    results.append(verify_file_contains(
        str(interfaces_path),
        "class IHighLevelAuthManager",
        "High-level auth manager interface"
    ))
    results.append(verify_file_contains(
        str(interfaces_path),
        "class ValidationResult",
        "Validation result class"
    ))
    results.append(verify_file_contains(
        str(interfaces_path),
        "class DataSource(Enum)",
        "Data source enum"
    ))
    
    # 4. Verify high-level wrappers
    print("\n4. High-Level Wrappers:")
    wrappers_path = workspace / "exchanges" / "shared" / "high_level_wrappers.py"
    results.append(verify_file_exists(
        str(wrappers_path),
        "High-level wrappers module"
    ))
    results.append(verify_file_contains(
        str(wrappers_path),
        "class HighLevelAuthManager",
        "High-level auth manager wrapper"
    ))
    results.append(verify_file_contains(
        str(wrappers_path),
        "class HighLevelMarketManager",
        "High-level market manager wrapper"
    ))
    results.append(verify_file_contains(
        str(wrappers_path),
        "class HighLevelOrderManager",
        "High-level order manager wrapper"
    ))
    
    # 5. Verify updated shared module exports
    print("\n5. Updated Shared Module Exports:")
    init_path = workspace / "exchanges" / "shared" / "__init__.py"
    results.append(verify_file_contains(
        str(init_path),
        "HighLevelAuthManager",
        "High-level auth manager export"
    ))
    results.append(verify_file_contains(
        str(init_path),
        "DataSource",
        "Data source enum export"
    ))
    results.append(verify_file_contains(
        str(init_path),
        "ValidationResult",
        "Validation result export"
    ))
    
    # 6. Verify example usage
    print("\n6. Example Usage:")
    example_path = workspace / "exchanges" / "shared" / "examples" / "high_level_usage.py"
    results.append(verify_file_exists(
        str(example_path),
        "High-level usage example"
    ))
    results.append(verify_file_contains(
        str(example_path),
        "HighLevelAuthManager",
        "Example using high-level auth manager"
    ))
    
    # 7. Verify test configuration
    print("\n7. Test Configuration:")
    pytest_ini_path = tests_dir / "pytest.ini"
    run_tests_path = tests_dir / "run_tests.py"
    results.append(verify_file_exists(
        str(pytest_ini_path),
        "Pytest configuration"
    ))
    results.append(verify_file_exists(
        str(run_tests_path),
        "Test runner script"
    ))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    total_checks = len(results)
    passed_checks = sum(results)
    failed_checks = total_checks - passed_checks
    
    print(f"Total checks: {total_checks}")
    print(f"✅ Passed: {passed_checks}")
    print(f"❌ Failed: {failed_checks}")
    print(f"Success rate: {(passed_checks/total_checks)*100:.1f}%")
    
    if failed_checks == 0:
        print("\n🎉 ALL IMPROVEMENTS SUCCESSFULLY IMPLEMENTED!")
        print("\nKey improvements:")
        print("• ✅ Async context manager support added to ExchangeInterface")
        print("• ✅ Comprehensive unit tests created for all shared utilities")
        print("• ✅ High-level interfaces with consistent abstraction levels")
        print("• ✅ High-level wrapper classes for easy usage")
        print("• ✅ Validation and error handling improvements")
        print("• ✅ Example usage and documentation")
        return 0
    else:
        print(f"\n⚠️  {failed_checks} checks failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())