"""
Simple test to verify the improvements work.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Test the ExchangeInterface improvements
def test_exchange_interface_async_context_manager():
    """Test that ExchangeInterface has async context manager support."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from exchanges.base_exchange.exchange_interface import IExchange, ExchangeStatus, OrderSide, OrderType
    
    # Check that the interface has the required methods
    assert hasattr(IExchange, '__aenter__')
    assert hasattr(IExchange, '__aexit__')
    
    # Check method signatures
    import inspect
    aenter_sig = inspect.signature(IExchange.__aenter__)
    aexit_sig = inspect.signature(IExchange.__aexit__)
    
    assert len(aenter_sig.parameters) == 1  # self
    assert len(aexit_sig.parameters) == 4  # self, exc_type, exc_val, exc_tb


def test_high_level_interfaces():
    """Test that high-level interfaces exist and have correct methods."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from exchanges.shared.interfaces import (
        DataSource, ValidationResult, IHighLevelAuthManager,
        IHighLevelMarketManager, IHighLevelOrderManager
    )
    
    # Test enums
    assert DataSource.CACHE.value == "cache"
    assert DataSource.EXCHANGE.value == "exchange"
    assert DataSource.FALLBACK.value == "fallback"
    
    # Test ValidationResult
    result = ValidationResult(True)
    assert result.is_valid is True
    assert result.errors == []
    assert result.warnings == []
    
    result.add_error("Test error")
    assert result.is_valid is False
    assert "Test error" in result.errors
    
    result.add_warning("Test warning")
    assert "Test warning" in result.warnings
    
    # Test that interfaces have required methods
    assert hasattr(IHighLevelAuthManager, 'authenticate')
    assert hasattr(IHighLevelAuthManager, 'is_authenticated')
    assert hasattr(IHighLevelMarketManager, 'get_price')
    assert hasattr(IHighLevelOrderManager, 'create_order')


def test_high_level_wrappers():
    """Test that high-level wrappers can be instantiated."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from exchanges.shared.high_level_wrappers import (
        HighLevelAuthManager,
        HighLevelMarketManager,
        HighLevelOrderManager,
        HighLevelRiskManager,
        HighLevelBalanceManager,
        HighLevelRateLimitManager
    )
    
    # Test instantiation
    auth_manager = HighLevelAuthManager("test")
    market_manager = HighLevelMarketManager("test")
    order_manager = HighLevelOrderManager("test")
    risk_manager = HighLevelRiskManager("test")
    balance_manager = HighLevelBalanceManager("test")
    rate_limit_manager = HighLevelRateLimitManager("test")
    
    # Test initialization
    auth_manager.initialize()
    market_manager.initialize()
    order_manager.initialize()
    risk_manager.initialize()
    balance_manager.initialize()
    rate_limit_manager.initialize()
    
    # Test status
    assert auth_manager.get_status()["initialized"] is True
    assert market_manager.get_status()["initialized"] is True
    assert order_manager.get_status()["initialized"] is True
    assert risk_manager.get_status()["initialized"] is True
    assert balance_manager.get_status()["initialized"] is True
    assert rate_limit_manager.get_status()["initialized"] is True
    
    # Test reset
    auth_manager.reset()
    market_manager.reset()
    order_manager.reset()
    risk_manager.reset()
    balance_manager.reset()
    rate_limit_manager.reset()
    
    # Test close
    auth_manager.close()
    market_manager.close()
    order_manager.close()
    risk_manager.close()
    balance_manager.close()
    rate_limit_manager.close()


@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality of high-level wrappers."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from exchanges.shared.high_level_wrappers import HighLevelAuthManager
    
    auth_manager = HighLevelAuthManager("test")
    auth_manager.initialize()
    
    # Test authentication with mock credentials
    credentials = {
        "api_key": "test_key",
        "api_secret": "test_secret",
        "permissions": ["read"]
    }
    
    # This should work even without actual exchange connection
    # since we're testing the interface
    try:
        result = await auth_manager.authenticate(credentials)
        # Result might be False due to no actual exchange connection
        assert isinstance(result, bool)
    except Exception as e:
        # Expected to fail without actual exchange setup
        assert "test" in str(e).lower() or "connection" in str(e).lower()
    
    auth_manager.close()


def test_validation_result():
    """Test ValidationResult class functionality."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from exchanges.shared.interfaces import ValidationResult
    
    # Test valid result
    result = ValidationResult(True)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert len(result.warnings) == 0
    
    # Test adding errors
    result.add_error("Error 1")
    result.add_error("Error 2")
    assert result.is_valid is False
    assert len(result.errors) == 2
    assert "Error 1" in result.errors
    assert "Error 2" in result.errors
    
    # Test adding warnings
    result.add_warning("Warning 1")
    result.add_warning("Warning 2")
    assert len(result.warnings) == 2
    assert "Warning 1" in result.warnings
    assert "Warning 2" in result.warnings
    
    # Test invalid result
    result = ValidationResult(False, ["Initial error"], ["Initial warning"])
    assert result.is_valid is False
    assert len(result.errors) == 1
    assert len(result.warnings) == 1


def test_data_source_enum():
    """Test DataSource enum functionality."""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from exchanges.shared.interfaces import DataSource
    
    # Test enum values
    assert DataSource.CACHE.value == "cache"
    assert DataSource.EXCHANGE.value == "exchange"
    assert DataSource.FALLBACK.value == "fallback"
    
    # Test enum iteration
    sources = list(DataSource)
    assert len(sources) == 3
    assert DataSource.CACHE in sources
    assert DataSource.EXCHANGE in sources
    assert DataSource.FALLBACK in sources


if __name__ == "__main__":
    # Run tests directly
    test_exchange_interface_async_context_manager()
    test_high_level_interfaces()
    test_high_level_wrappers()
    test_validation_result()
    test_data_source_enum()
    
    # Run async test
    asyncio.run(test_async_functionality())
    
    print("✅ All tests passed!")