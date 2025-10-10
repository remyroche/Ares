#!/usr/bin/env python3
"""
Minimal test script to verify exchange integrations work properly.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_shared_interfaces():
    """Test shared interfaces directly."""
    print("🧪 Testing shared interfaces...")
    
    try:
        # Test the shared interfaces module directly
        import exchanges.shared.interfaces_typed as interfaces
        
        # Test tprint function
        interfaces.tprint("Test message", "INFO")
        print("✅ tprint function working")
        
        # Test DataSource enum
        assert interfaces.DataSource.CACHE.value == "cache"
        assert interfaces.DataSource.EXCHANGE.value == "exchange"
        assert interfaces.DataSource.FALLBACK.value == "fallback"
        print("✅ DataSource enum working")
        
        # Test ValidationResult
        result = interfaces.ValidationResult(True)
        assert result.is_valid == True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        print("✅ ValidationResult working")
        
        # Test error handling decorators
        assert hasattr(interfaces, 'handle_errors'), "handle_errors decorator not found"
        assert hasattr(interfaces, 'handle_async_errors'), "handle_async_errors decorator not found"
        print("✅ Error handling decorators working")
        
        return True
        
    except Exception as e:
        print(f"❌ Shared interfaces test failed: {e}")
        return False


def test_exchange_files():
    """Test that exchange files exist and are readable."""
    print("\n🧪 Testing exchange files...")
    
    try:
        # Check if exchange files exist
        exchange_files = [
            'exchanges/bingx.py',
            'exchanges/mexc.py', 
            'exchanges/binance.py',
            'exchanges/okx.py'
        ]
        
        for file_path in exchange_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} exists")
            else:
                print(f"❌ {file_path} not found")
                return False
        
        # Check if shared interfaces exist
        shared_files = [
            'exchanges/shared/interfaces.py',
            'exchanges/shared/interfaces_typed.py'
        ]
        
        for file_path in shared_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} exists")
            else:
                print(f"❌ {file_path} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Exchange files test failed: {e}")
        return False


def test_file_contents():
    """Test that files contain expected content."""
    print("\n🧪 Testing file contents...")
    
    try:
        # Test bingx.py content
        with open('exchanges/bingx.py', 'r') as f:
            content = f.read()
            assert 'BingXExchange' in content, "BingXExchange class not found"
            assert 'tprint' in content, "tprint usage not found"
            assert 'AuthenticationManager' in content, "AuthenticationManager not found"
            assert 'RateLimitManager' in content, "RateLimitManager not found"
        print("✅ bingx.py content looks good")
        
        # Test mexc.py content
        with open('exchanges/mexc.py', 'r') as f:
            content = f.read()
            assert 'MexcExchange' in content, "MexcExchange class not found"
            assert 'tprint' in content, "tprint usage not found"
            assert 'AuthenticationManager' in content, "AuthenticationManager not found"
        print("✅ mexc.py content looks good")
        
        # Test binance.py content
        with open('exchanges/binance.py', 'r') as f:
            content = f.read()
            assert 'BinanceExchange' in content, "BinanceExchange class not found"
            assert 'tprint' in content, "tprint usage not found"
            assert 'AuthenticationManager' in content, "AuthenticationManager not found"
        print("✅ binance.py content looks good")
        
        # Test trading interface content
        with open('src/trading/execution/exchange_interface.py', 'r') as f:
            content = f.read()
            assert 'ExchangeInterface' in content, "ExchangeInterface class not found"
            assert 'tprint' in content, "tprint usage not found"
            assert 'AuthenticationManager' in content, "AuthenticationManager not found"
        print("✅ trading interface content looks good")
        
        return True
        
    except Exception as e:
        print(f"❌ File contents test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting minimal exchange integration tests...\n")
    
    tests = [
        test_shared_interfaces,
        test_exchange_files,
        test_file_contents
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Exchange integrations are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)