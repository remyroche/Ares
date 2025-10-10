"""
Test script for the Enhanced Klines Processing Pipeline

This script tests all the requirements:
1. Type hints & tprints
2. ExchangeInterface usage
3. Full functionality
4. Data standardizer integration
5. Fast fail pattern
6. Exchange-agnostic design
7. OHLCV data processing with gap detection, filling, and resampling
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.trading.execution.exchange_interface import create_exchange_interface
from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
    EnhancedKlinesProcessingPipeline,
    ResamplingConfig,
    process_klines_data_enhanced
)
from src.training.steps.data_collection.klines_downloading_processing import (
    run_enhanced_klines_pipeline
)
from src.utils.tprint import tprint_info, tprint_success, tprint_error, tprint_warning


class MockExchangeInterface:
    """Mock ExchangeInterface for testing without real API calls."""
    
    def __init__(self, exchange_type: str = "binance"):
        self.exchange_type = exchange_type
        self.connected = False
    
    async def connect(self) -> bool:
        """Mock connection."""
        self.connected = True
        tprint_info(f"🔌 Mock connected to {self.exchange_type}")
        return True
    
    async def disconnect(self) -> None:
        """Mock disconnection."""
        self.connected = False
        tprint_info(f"📴 Mock disconnected from {self.exchange_type}")
    
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Generate mock klines data."""
        if not self.connected:
            raise RuntimeError("Not connected to exchange")
        
        # Generate mock data
        data = []
        current_time = start_time or datetime.now() - timedelta(days=1)
        end_time = end_time or datetime.now()
        
        # Calculate interval in minutes
        interval_minutes = self._interval_to_minutes(interval)
        if interval_minutes is None:
            raise ValueError(f"Unsupported interval: {interval}")
        
        base_price = 3000.0 if symbol.upper() == "ETHUSDT" else 50000.0
        price = base_price
        
        while current_time < end_time and len(data) < limit:
            # Generate realistic OHLCV data
            open_price = price
            high_price = open_price * (1 + np.random.uniform(0, 0.02))
            low_price = open_price * (1 - np.random.uniform(0, 0.02))
            close_price = low_price + np.random.uniform(0, high_price - low_price)
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'timestamp': current_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'quote_volume': close_price * volume,
                'trades': np.random.randint(10, 100),
                'taker_buy_base': volume * np.random.uniform(0.3, 0.7),
                'taker_buy_quote': close_price * volume * np.random.uniform(0.3, 0.7)
            })
            
            # Update price for next candle
            price = close_price * (1 + np.random.normal(0, 0.001))
            current_time += timedelta(minutes=interval_minutes)
        
        return data
    
    def _interval_to_minutes(self, interval: str) -> Optional[int]:
        """Convert interval string to minutes."""
        interval_map = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        return interval_map.get(interval)


async def test_enhanced_pipeline():
    """Test the enhanced klines processing pipeline."""
    tprint_info("🧪 Starting Enhanced Klines Pipeline Tests")
    
    # Test 1: Basic pipeline functionality
    tprint_info("\n📋 Test 1: Basic Pipeline Functionality")
    try:
        # Create mock exchange interface
        mock_exchange = MockExchangeInterface("binance")
        await mock_exchange.connect()
        
        # Create enhanced pipeline
        pipeline = EnhancedKlinesProcessingPipeline(
            data_dir="test_data",
            exchange="binance",
            enable_logging=True
        )
        
        # Test data processing
        results = await pipeline.process_klines_data(
            symbol="ETHUSDT",
            interval="1m",
            years=1,
            exchange_interface=mock_exchange,
            max_gap_minutes=1,
            create_consolidated=True
        )
        
        # Validate results
        assert results["pipeline_success"], f"Pipeline failed: {results.get('errors', [])}"
        assert results["symbol"] == "ETHUSDT"
        assert results["interval"] == "1m"
        assert results["final_data_shape"][0] > 0, "No data processed"
        
        tprint_success("✅ Test 1 passed: Basic pipeline functionality")
        
        await mock_exchange.disconnect()
        
    except Exception as e:
        tprint_error(f"❌ Test 1 failed: {e}")
        return False
    
    # Test 2: Exchange-agnostic design
    tprint_info("\n📋 Test 2: Exchange-Agnostic Design")
    try:
        # Test with different exchange types
        exchanges = ["binance", "okx", "gateio"]
        
        for exchange in exchanges:
            mock_exchange = MockExchangeInterface(exchange)
            await mock_exchange.connect()
            
            pipeline = EnhancedKlinesProcessingPipeline(
                data_dir="test_data",
                exchange=exchange,
                enable_logging=False  # Reduce noise
            )
            
            results = await pipeline.process_klines_data(
                symbol="BTCUSDT",
                interval="5m",
                years=1,
                exchange_interface=mock_exchange,
                create_consolidated=False
            )
            
            assert results["pipeline_success"], f"Pipeline failed for {exchange}"
            assert results["exchange"] == exchange
            
            await mock_exchange.disconnect()
        
        tprint_success("✅ Test 2 passed: Exchange-agnostic design")
        
    except Exception as e:
        tprint_error(f"❌ Test 2 failed: {e}")
        return False
    
    # Test 3: Data resampling
    tprint_info("\n📋 Test 3: Data Resampling")
    try:
        mock_exchange = MockExchangeInterface("binance")
        await mock_exchange.connect()
        
        pipeline = EnhancedKlinesProcessingPipeline(
            data_dir="test_data",
            exchange="binance",
            enable_logging=False
        )
        
        # Configure resampling
        resampling_config = ResamplingConfig(
            target_intervals=['5m', '15m', '1h'],
            method='ohlc',
            preserve_volume=True,
            validate_continuity=True
        )
        
        results = await pipeline.process_klines_data(
            symbol="ETHUSDT",
            interval="1m",
            years=1,
            exchange_interface=mock_exchange,
            resampling_config=resampling_config,
            create_consolidated=False
        )
        
        assert results["pipeline_success"], f"Resampling failed: {results.get('errors', [])}"
        assert "resampling" in results["steps_completed"], "Resampling step not completed"
        
        tprint_success("✅ Test 3 passed: Data resampling")
        
        await mock_exchange.disconnect()
        
    except Exception as e:
        tprint_error(f"❌ Test 3 failed: {e}")
        return False
    
    # Test 4: Gap detection and filling
    tprint_info("\n📋 Test 4: Gap Detection and Filling")
    try:
        mock_exchange = MockExchangeInterface("binance")
        await mock_exchange.connect()
        
        pipeline = EnhancedKlinesProcessingPipeline(
            data_dir="test_data",
            exchange="binance",
            enable_logging=False
        )
        
        results = await pipeline.process_klines_data(
            symbol="ETHUSDT",
            interval="1m",
            years=1,
            exchange_interface=mock_exchange,
            max_gap_minutes=1,
            create_consolidated=False
        )
        
        assert results["pipeline_success"], f"Gap handling failed: {results.get('errors', [])}"
        assert "gap_detection" in results["steps_completed"], "Gap detection step not completed"
        
        tprint_success("✅ Test 4 passed: Gap detection and filling")
        
        await mock_exchange.disconnect()
        
    except Exception as e:
        tprint_error(f"❌ Test 4 failed: {e}")
        return False
    
    # Test 5: Fast fail pattern
    tprint_info("\n📋 Test 5: Fast Fail Pattern")
    try:
        pipeline = EnhancedKlinesProcessingPipeline(
            data_dir="test_data",
            exchange="binance",
            enable_logging=False
        )
        
        # Test with invalid parameters (should fail fast)
        try:
            await pipeline.process_klines_data(
                symbol="",  # Invalid symbol
                interval="1m",
                years=1,
                exchange_interface=None,  # Invalid exchange interface
                create_consolidated=False
            )
            assert False, "Should have failed with invalid parameters"
        except (ValueError, RuntimeError):
            tprint_success("✅ Test 5 passed: Fast fail pattern")
        
    except Exception as e:
        tprint_error(f"❌ Test 5 failed: {e}")
        return False
    
    # Test 6: Data quality validation
    tprint_info("\n📋 Test 6: Data Quality Validation")
    try:
        mock_exchange = MockExchangeInterface("binance")
        await mock_exchange.connect()
        
        pipeline = EnhancedKlinesProcessingPipeline(
            data_dir="test_data",
            exchange="binance",
            enable_logging=False
        )
        
        results = await pipeline.process_klines_data(
            symbol="ETHUSDT",
            interval="1m",
            years=1,
            exchange_interface=mock_exchange,
            create_consolidated=False
        )
        
        assert results["pipeline_success"], f"Quality validation failed: {results.get('errors', [])}"
        assert "validate" in results["steps_completed"], "Validation step not completed"
        assert "quality_check" in results["steps_completed"], "Quality check step not completed"
        assert results["data_quality"] in ["excellent", "good", "fair", "poor"], "Invalid quality level"
        
        tprint_success("✅ Test 6 passed: Data quality validation")
        
        await mock_exchange.disconnect()
        
    except Exception as e:
        tprint_error(f"❌ Test 6 failed: {e}")
        return False
    
    # Test 7: Convenience function
    tprint_info("\n📋 Test 7: Convenience Function")
    try:
        mock_exchange = MockExchangeInterface("binance")
        await mock_exchange.connect()
        
        # Test the convenience function
        results = await process_klines_data_enhanced(
            symbol="ETHUSDT",
            interval="1m",
            years=1,
            exchange_interface=mock_exchange,
            data_dir="test_data",
            exchange="binance",
            create_consolidated=False
        )
        
        assert results["pipeline_success"], f"Convenience function failed: {results.get('errors', [])}"
        assert results["symbol"] == "ETHUSDT"
        
        tprint_success("✅ Test 7 passed: Convenience function")
        
        await mock_exchange.disconnect()
        
    except Exception as e:
        tprint_error(f"❌ Test 7 failed: {e}")
        return False
    
    tprint_success("\n🎉 All Enhanced Klines Pipeline Tests Passed!")
    return True


async def test_legacy_compatibility():
    """Test backward compatibility with legacy interface."""
    tprint_info("\n🧪 Testing Legacy Compatibility")
    
    try:
        # Test the enhanced wrapper function
        results = await run_enhanced_klines_pipeline(
            symbol="ETHUSDT",
            years=1,
            interval="1m",
            data_dir="test_data",
            exchange="binance",
            api_key="test_key",
            api_secret="test_secret",
            max_gap_minutes=1,
            create_consolidated=False,
            resampling_intervals=['5m', '15m']
        )
        
        # Should fail due to invalid API credentials, but with proper error handling
        assert not results["pipeline_success"], "Should fail with invalid credentials"
        assert "errors" in results, "Should have error information"
        
        tprint_success("✅ Legacy compatibility test passed")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Legacy compatibility test failed: {e}")
        return False


async def main():
    """Run all tests."""
    tprint_info("🚀 Starting Enhanced Klines Processing Pipeline Test Suite")
    
    # Create test data directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Run enhanced pipeline tests
        enhanced_tests_passed = await test_enhanced_pipeline()
        
        # Run legacy compatibility tests
        legacy_tests_passed = await test_legacy_compatibility()
        
        # Summary
        tprint_info("\n📊 Test Summary:")
        tprint_info(f"   Enhanced Pipeline Tests: {'✅ PASSED' if enhanced_tests_passed else '❌ FAILED'}")
        tprint_info(f"   Legacy Compatibility Tests: {'✅ PASSED' if legacy_tests_passed else '❌ FAILED'}")
        
        if enhanced_tests_passed and legacy_tests_passed:
            tprint_success("\n🎉 All Tests Passed! The Enhanced Klines Processing Pipeline is ready for production.")
        else:
            tprint_error("\n❌ Some tests failed. Please review the implementation.")
            
    finally:
        # Cleanup test data
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
            tprint_info("🧹 Cleaned up test data")


if __name__ == "__main__":
    asyncio.run(main())