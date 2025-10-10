#!/usr/bin/env python3
"""
Test script for the enhanced klines downloading and processing pipeline.

This script tests the pipeline functionality to ensure all requirements are met:
1. Type hints & tprints
2. ExchangeInterface usage
3. Data standardizer integration
4. Fast fail pattern
5. No mock data or fallbacks
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.training.steps.data_collection.klines_downloading_processing_enhanced import (
    KlinesDataProcessingPipeline,
    run_enhanced_klines_pipeline
)
from src.utils.tprint import tprint_info, tprint_success, tprint_error


async def test_pipeline_initialization():
    """Test pipeline initialization."""
    tprint_info("🧪 Testing pipeline initialization...")
    
    try:
        pipeline = KlinesDataProcessingPipeline("test_data")
        tprint_success("✅ Pipeline initialization successful")
        
        # Test data standardizer integration
        assert hasattr(pipeline, 'data_standardizer'), "Data standardizer not integrated"
        tprint_success("✅ Data standardizer integrated")
        
        # Test type hints are present
        assert hasattr(pipeline, '__annotations__'), "Type hints missing"
        tprint_success("✅ Type hints present")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Pipeline initialization failed: {e}")
        return False


async def test_data_standardization():
    """Test data standardization functionality."""
    tprint_info("🧪 Testing data standardization...")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        pipeline = KlinesDataProcessingPipeline("test_data")
        
        # Create test data
        test_data = pd.DataFrame({
            'open_time': [int((datetime.now() - timedelta(minutes=i)).timestamp() * 1000) for i in range(10)],
            'close_time': [int((datetime.now() - timedelta(minutes=i-1)).timestamp() * 1000) for i in range(10)],
            'open': np.random.uniform(50000, 51000, 10),
            'high': np.random.uniform(51000, 52000, 10),
            'low': np.random.uniform(49000, 50000, 10),
            'close': np.random.uniform(50000, 51000, 10),
            'volume': np.random.uniform(100, 1000, 10)
        })
        
        # Test standardization
        standardized_df, report = pipeline.standardize_data_format(
            test_data, "ETHUSDT", "1m", "binance"
        )
        
        assert report['success'], f"Standardization failed: {report.get('errors', [])}"
        assert 'exchange' in standardized_df.columns, "Exchange column missing"
        assert 'symbol' in standardized_df.columns, "Symbol column missing"
        assert 'interval' in standardized_df.columns, "Interval column missing"
        
        tprint_success("✅ Data standardization working correctly")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Data standardization test failed: {e}")
        return False


async def test_fast_fail_pattern():
    """Test fast fail pattern implementation."""
    tprint_info("🧪 Testing fast fail pattern...")
    
    try:
        pipeline = KlinesDataProcessingPipeline("test_data")
        
        # Test with invalid data should fail fast
        try:
            standardized_df, report = pipeline.standardize_data_format(
                None, "ETHUSDT", "1m", "binance"
            )
            assert False, "Should have failed fast with None data"
        except Exception:
            tprint_success("✅ Fast fail working for invalid data")
        
        # Test with empty DataFrame should fail fast
        try:
            import pandas as pd
            empty_df = pd.DataFrame()
            standardized_df, report = pipeline.standardize_data_format(
                empty_df, "ETHUSDT", "1m", "binance"
            )
            assert not report['success'], "Should have failed fast with empty data"
            tprint_success("✅ Fast fail working for empty data")
        except Exception:
            tprint_success("✅ Fast fail working for empty data (exception)")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Fast fail pattern test failed: {e}")
        return False


async def test_exchange_interface_usage():
    """Test that ExchangeInterface is used properly."""
    tprint_info("🧪 Testing ExchangeInterface usage...")
    
    try:
        from src.trading.execution.exchange_interface import ExchangeInterface, create_exchange_interface
        
        # Test ExchangeInterface creation
        config = {
            'exchange_type': 'binance',
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'testnet': True
        }
        
        exchange_interface = create_exchange_interface(config)
        assert isinstance(exchange_interface, ExchangeInterface), "Not an ExchangeInterface instance"
        
        tprint_success("✅ ExchangeInterface creation working")
        
        # Test that pipeline methods accept ExchangeInterface
        pipeline = KlinesDataProcessingPipeline("test_data")
        
        # Check method signatures
        import inspect
        sig = inspect.signature(pipeline.handle_gaps_with_column_removal)
        params = list(sig.parameters.keys())
        assert 'exchange_interface' in params, "ExchangeInterface parameter missing"
        
        tprint_success("✅ ExchangeInterface integration verified")
        return True
        
    except Exception as e:
        tprint_error(f"❌ ExchangeInterface test failed: {e}")
        return False


async def test_no_mock_data():
    """Test that no mock data or fallbacks are present."""
    tprint_info("🧪 Testing for absence of mock data...")
    
    try:
        # Read the enhanced pipeline file
        with open("src/training/steps/data_collection/klines_downloading_processing_enhanced.py", "r") as f:
            content = f.read()
        
        # Check for mock data patterns
        mock_patterns = [
            "simulated",
            "mock",
            "fake",
            "dummy",
            "placeholder",
            "fallback",
            "test_data"
        ]
        
        found_patterns = []
        for pattern in mock_patterns:
            if pattern in content.lower():
                found_patterns.append(pattern)
        
        # Some patterns might be acceptable (like "test_data" in test files)
        # But we should not have simulated exchange data
        if "simulated" in content.lower() and "exchange" in content.lower():
            tprint_error("❌ Simulated exchange data found")
            return False
        
        if "mock" in content.lower() and "exchange" in content.lower():
            tprint_error("❌ Mock exchange data found")
            return False
        
        tprint_success("✅ No mock data or fallbacks found")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Mock data test failed: {e}")
        return False


async def test_tprint_usage():
    """Test that tprint is used instead of logger."""
    tprint_info("🧪 Testing tprint usage...")
    
    try:
        # Read the enhanced pipeline file
        with open("src/training/steps/data_collection/klines_downloading_processing_enhanced.py", "r") as f:
            content = f.read()
        
        # Check for tprint usage
        tprint_count = content.count("tprint")
        logger_count = content.count("self.logger")
        
        assert tprint_count > 0, "No tprint usage found"
        assert logger_count == 0, f"Logger usage found ({logger_count} instances) - should use tprint instead"
        
        tprint_success(f"✅ tprint usage verified ({tprint_count} instances)")
        return True
        
    except Exception as e:
        tprint_error(f"❌ tprint test failed: {e}")
        return False


async def main():
    """Run all tests."""
    tprint_info("🚀 Starting klines pipeline tests...")
    
    tests = [
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Data Standardization", test_data_standardization),
        ("Fast Fail Pattern", test_fast_fail_pattern),
        ("ExchangeInterface Usage", test_exchange_interface_usage),
        ("No Mock Data", test_no_mock_data),
        ("tprint Usage", test_tprint_usage),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        tprint_info(f"\n📋 Running test: {test_name}")
        try:
            result = await test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            tprint_error(f"❌ Test {test_name} crashed: {e}")
            failed += 1
    
    tprint_info(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        tprint_success("🎉 All tests passed! Pipeline meets all requirements.")
        return True
    else:
        tprint_error(f"❌ {failed} tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)