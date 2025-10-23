#!/usr/bin/env python3
"""
VectorBT installation validation script.

This script validates that VectorBT is properly installed and configured
for production use in the Ares trading system.
"""

import sys
import logging
from pathlib import Path

# Add workspace root to Python path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_vectorbt_import():
    """Validate VectorBT can be imported."""
    try:
        from src.vectorbt import (
            vbt, rolling_mean, rolling_std, Portfolio, PortfolioFactory,
            RSI, MACD, BBANDS, validate_vectorbt_installation, get_vectorbt_info
        )
        logger.info("✅ VectorBT imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ VectorBT import failed: {e}")
        return False

def validate_basic_functionality():
    """Validate basic VectorBT functionality."""
    try:
        import pandas as pd
        import numpy as np
        from src.vectorbt import rolling_mean, rolling_std, PortfolioFactory
        
        # Create test data
        data = pd.Series(np.random.randn(100))
        
        # Test rolling operations
        rolling_mean(data, window=10)
        rolling_std(data, window=10)
        
        # Test portfolio creation
        returns = data.pct_change().dropna()
        portfolio = PortfolioFactory.from_returns(returns)
        
        logger.info("✅ Basic functionality test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False

def validate_technical_indicators():
    """Validate technical indicators."""
    try:
        import pandas as pd
        import numpy as np
        from src.vectorbt import RSI, MACD, BBANDS
        
        # Create test data
        data = pd.Series(np.random.randn(100))
        
        # Test indicators
        rsi = RSI.run(data)
        macd = MACD.run(data)
        bbands = BBANDS.run(data)
        
        logger.info("✅ Technical indicators test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Technical indicators test failed: {e}")
        return False

def validate_performance_monitoring():
    """Validate performance monitoring."""
    try:
        from src.vectorbt import monitor_operation, get_performance_monitor
        import pandas as pd
        import numpy as np
        
        # Test performance monitoring
        data = pd.Series(np.random.randn(100))
        
        with monitor_operation("test_operation", data_size=len(data)):
            result = data.rolling(10).mean()
        
        monitor = get_performance_monitor()
        stats = monitor.get_operation_stats("test_operation")
        
        logger.info("✅ Performance monitoring test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Performance monitoring test failed: {e}")
        return False

def validate_configuration():
    """Validate configuration system."""
    try:
        from src.vectorbt import (
            VectorBTConfig, configure_vectorbt, get_vectorbt_config,
            PRODUCTION_CONFIG, DEVELOPMENT_CONFIG
        )
        
        # Test configuration
        config = get_vectorbt_config()
        assert isinstance(config, VectorBTConfig)
        
        # Test production config
        assert PRODUCTION_CONFIG.production_mode is True
        assert PRODUCTION_CONFIG.memory_efficient is True
        
        # Test development config
        assert DEVELOPMENT_CONFIG.debug_mode is True
        
        logger.info("✅ Configuration test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def validate_error_handling():
    """Validate error handling."""
    try:
        from src.vectorbt import (
            VectorBTError, VectorBTDataError, VectorBTComputationError,
            ProductionPortfolioFactory
        )
        import pandas as pd
        
        # Test data error
        try:
            ProductionPortfolioFactory.from_returns("invalid_data")
            assert False, "Should have raised VectorBTDataError"
        except VectorBTDataError:
            pass  # Expected
        
        # Test computation error
        try:
            ProductionPortfolioFactory.from_returns(pd.Series([np.nan] * 100))
            assert False, "Should have raised VectorBTComputationError"
        except VectorBTComputationError:
            pass  # Expected
        
        logger.info("✅ Error handling test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    logger.info("🔍 Starting VectorBT installation validation")
    
    tests = [
        ("Import Validation", validate_vectorbt_import),
        ("Basic Functionality", validate_basic_functionality),
        ("Technical Indicators", validate_technical_indicators),
        ("Performance Monitoring", validate_performance_monitoring),
        ("Configuration", validate_configuration),
        ("Error Handling", validate_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name}...")
        if test_func():
            passed += 1
        else:
            logger.error(f"❌ {test_name} failed")
    
    logger.info(f"📊 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 VectorBT installation validation successful!")
        logger.info("VectorBT is ready for production use")
        return True
    else:
        logger.error("❌ VectorBT installation validation failed")
        logger.error("Please check the installation and configuration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)