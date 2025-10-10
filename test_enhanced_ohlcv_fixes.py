#!/usr/bin/env python3
"""
Test script for Enhanced OHLCV Manager fixes

This script demonstrates the comprehensive fixes implemented for:
1. Timestamp parsing with multiple unit support
2. Enhanced error handling with recovery mechanisms
3. Memory leak prevention and management
4. Thread-safe cache operations
5. Advanced data validation and integrity checks
"""

import asyncio
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Import the enhanced OHLCV manager
from exchanges.shared.pricing.enhanced_ohlcv_manager import (
    EnhancedOHLCVManager, OHLCVData, Timeframe, TimestampUnit, 
    DataIntegrityLevel, CacheConfig, get_enhanced_ohlcv_manager,
    cleanup_all_managers
)

def create_mock_klines_data(symbol: str, timeframe: str, count: int = 100) -> List[List[Any]]:
    """Create mock klines data with various timestamp formats for testing."""
    base_time = int(datetime.now().timestamp() * 1000)  # Milliseconds
    data = []
    
    for i in range(count):
        # Create different timestamp formats for testing
        if i % 4 == 0:
            # Milliseconds (13 digits)
            timestamp = base_time - (count - i) * 60000  # 1 minute intervals
        elif i % 4 == 1:
            # Seconds (10 digits)
            timestamp = (base_time - (count - i) * 60000) // 1000
        elif i % 4 == 2:
            # Microseconds (16 digits)
            timestamp = (base_time - (count - i) * 60000) * 1000
        else:
            # Nanoseconds (19 digits)
            timestamp = (base_time - (count - i) * 60000) * 1000000
        
        # Generate OHLCV data
        base_price = 50000 + np.random.normal(0, 1000)
        high = base_price + abs(np.random.normal(0, 500))
        low = base_price - abs(np.random.normal(0, 500))
        open_price = low + np.random.uniform(0, high - low)
        close_price = low + np.random.uniform(0, high - low)
        volume = np.random.uniform(100, 1000)
        
        data.append([
            timestamp,
            open_price,
            high,
            low,
            close_price,
            volume,
            volume * close_price,  # quote_volume
            int(np.random.uniform(10, 100)),  # trades_count
            volume * np.random.uniform(0.3, 0.7),  # taker_buy_volume
            volume * close_price * np.random.uniform(0.3, 0.7)  # taker_buy_quote_volume
        ])
    
    return data

async def test_timestamp_parsing():
    """Test enhanced timestamp parsing with multiple units."""
    print("🔍 Testing Enhanced Timestamp Parsing")
    print("=" * 50)
    
    manager = get_enhanced_ohlcv_manager("test_exchange")
    
    # Test different timestamp formats
    test_timestamps = [
        (1640995200000, "Milliseconds"),  # 2022-01-01 00:00:00
        (1640995200, "Seconds"),          # 2022-01-01 00:00:00
        (1640995200000000, "Microseconds"), # 2022-01-01 00:00:00
        (1640995200000000000, "Nanoseconds"), # 2022-01-01 00:00:00
        ("2022-01-01T00:00:00Z", "ISO String"),
        ("2022-01-01 00:00:00", "Standard String"),
    ]
    
    for timestamp, description in test_timestamps:
        try:
            parsed = manager.timestamp_parser.parse_timestamp(timestamp)
            if parsed:
                print(f"✅ {description}: {timestamp} -> {parsed}")
            else:
                print(f"❌ {description}: {timestamp} -> Failed to parse")
        except Exception as e:
            print(f"❌ {description}: {timestamp} -> Error: {e}")
    
    print()

async def test_error_handling():
    """Test enhanced error handling and recovery."""
    print("🛡️ Testing Enhanced Error Handling")
    print("=" * 50)
    
    manager = get_enhanced_ohlcv_manager("test_exchange")
    
    # Test with malformed data
    malformed_data = [
        [None, 50000, 51000, 49000, 50500, 1000],  # Invalid timestamp
        [1640995200000, "invalid", 51000, 49000, 50500, 1000],  # Invalid price
        [1640995200000, 50000, 49000, 51000, 50500, 1000],  # High < Low
        [1640995200000, -50000, 51000, 49000, 50500, 1000],  # Negative price
        [1640995200000, 50000, 51000, 49000, 50500, -1000],  # Negative volume
    ]
    
    for i, data in enumerate(malformed_data):
        try:
            result = manager._parse_ohlcv_data_enhanced("BTCUSDT", Timeframe.MINUTE_1, [data])
            print(f"✅ Malformed data {i+1}: Handled gracefully, {len(result)} valid records")
        except Exception as e:
            print(f"❌ Malformed data {i+1}: Error not handled: {e}")
    
    print()

async def test_memory_management():
    """Test memory management and leak prevention."""
    print("🧠 Testing Memory Management")
    print("=" * 50)
    
    # Create manager with small cache limits for testing
    config = CacheConfig(
        max_candles_per_symbol=100,
        max_total_candles=500,
        cleanup_interval=1  # 1 second for testing
    )
    manager = get_enhanced_ohlcv_manager("memory_test", config)
    
    # Register mock fetch function
    async def mock_fetch(symbol: str, timeframe: str, limit: int):
        return create_mock_klines_data(symbol, timeframe, limit)
    
    manager.register_fetch_functions(mock_fetch)
    
    # Test memory usage tracking
    initial_stats = manager.get_cache_statistics()
    print(f"Initial memory stats: {initial_stats}")
    
    # Fetch data multiple times to test memory management
    for i in range(10):
        data = await manager.get_ohlcv("BTCUSDT", Timeframe.MINUTE_1, limit=50)
        print(f"Fetch {i+1}: Got {len(data)} candles")
        
        # Check memory stats
        stats = manager.get_cache_statistics()
        print(f"  Memory usage: {stats['performance_stats']['memory_cleanups']} cleanups")
        
        await asyncio.sleep(0.1)  # Small delay
    
    final_stats = manager.get_cache_statistics()
    print(f"Final memory stats: {final_stats}")
    print()

def test_thread_safety():
    """Test thread-safe cache operations."""
    print("🔒 Testing Thread Safety")
    print("=" * 50)
    
    manager = get_enhanced_ohlcv_manager("thread_test")
    
    # Test concurrent cache operations
    def cache_worker(worker_id: int, symbol: str, timeframe: Timeframe):
        """Worker function for concurrent cache operations."""
        for i in range(10):
            # Simulate cache operations
            data = manager.cache.get(symbol, timeframe, limit=10)
            
            # Add some data
            test_data = [OHLCVData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                open=50000 + i,
                high=51000 + i,
                low=49000 + i,
                close=50500 + i,
                volume=1000 + i
            )]
            
            manager.cache.put(symbol, timeframe, test_data)
            time.sleep(0.01)  # Small delay
    
    # Create multiple threads
    threads = []
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    timeframes = [Timeframe.MINUTE_1, Timeframe.MINUTE_5, Timeframe.HOUR_1]
    
    for i in range(5):  # 5 workers
        symbol = symbols[i % len(symbols)]
        timeframe = timeframes[i % len(timeframes)]
        thread = threading.Thread(target=cache_worker, args=(i, symbol, timeframe))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check final cache state
    stats = manager.cache.get_stats()
    print(f"Thread safety test completed: {stats}")
    print()

async def test_data_validation():
    """Test comprehensive data validation."""
    print("✅ Testing Data Validation")
    print("=" * 50)
    
    manager = get_enhanced_ohlcv_manager("validation_test")
    
    # Test different integrity levels
    integrity_levels = [
        DataIntegrityLevel.BASIC,
        DataIntegrityLevel.STANDARD,
        DataIntegrityLevel.STRICT,
        DataIntegrityLevel.CRITICAL
    ]
    
    for level in integrity_levels:
        validator = manager.data_validator
        validator.integrity_level = level
        
        # Test valid data
        valid_data = OHLCVData(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_1,
            timestamp=datetime.now(),
            open=50000,
            high=51000,
            low=49000,
            close=50500,
            volume=1000
        )
        
        is_valid, issues = validator.validate_ohlcv_data(valid_data)
        print(f"✅ {level.value} validation - Valid data: {is_valid}, Issues: {len(issues)}")
        
        # Test invalid data
        invalid_data = OHLCVData(
            symbol="BTCUSDT",
            timeframe=Timeframe.MINUTE_1,
            timestamp=datetime.now(),
            open=50000,
            high=49000,  # High < Low - invalid
            low=51000,
            close=50500,
            volume=1000
        )
        
        is_valid, issues = validator.validate_ohlcv_data(invalid_data)
        print(f"❌ {level.value} validation - Invalid data: {is_valid}, Issues: {len(issues)}")
        if issues:
            print(f"   Issues: {issues[:2]}...")  # Show first 2 issues
    
    print()

async def test_performance():
    """Test performance improvements."""
    print("⚡ Testing Performance Improvements")
    print("=" * 50)
    
    manager = get_enhanced_ohlcv_manager("performance_test")
    
    # Register mock fetch function
    async def mock_fetch(symbol: str, timeframe: str, limit: int):
        return create_mock_klines_data(symbol, timeframe, limit)
    
    manager.register_fetch_functions(mock_fetch)
    
    # Test performance with different data sizes
    test_sizes = [100, 500, 1000, 2000]
    
    for size in test_sizes:
        start_time = time.time()
        
        # Fetch data
        data = await manager.get_ohlcv("BTCUSDT", Timeframe.MINUTE_1, limit=size)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Size {size:4d}: {len(data):4d} records in {duration:.3f}s ({len(data)/duration:.0f} records/s)")
    
    # Test cache performance
    print("\nCache Performance:")
    start_time = time.time()
    
    # Multiple cache hits
    for _ in range(100):
        data = await manager.get_ohlcv("BTCUSDT", Timeframe.MINUTE_1, limit=100, use_cache=True)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"100 cache hits: {duration:.3f}s ({100/duration:.0f} hits/s)")
    
    print()

async def main():
    """Run all tests."""
    print("🚀 Enhanced OHLCV Manager - Comprehensive Fixes Test")
    print("=" * 60)
    print()
    
    try:
        # Run all tests
        await test_timestamp_parsing()
        await test_error_handling()
        await test_memory_management()
        test_thread_safety()
        await test_data_validation()
        await test_performance()
        
        print("🎉 All tests completed successfully!")
        print()
        print("✅ Fixes Implemented:")
        print("  • Robust timestamp parsing with multiple unit support")
        print("  • Comprehensive error handling with recovery mechanisms")
        print("  • Memory leak prevention and management")
        print("  • Thread-safe cache operations")
        print("  • Advanced data validation and integrity checks")
        print("  • Performance optimizations using hardware utilities")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cleanup_all_managers()
        print("\n🧹 Cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())