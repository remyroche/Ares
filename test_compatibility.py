#!/usr/bin/env python3
"""
Compatibility Test Script for ExchangeInterface, KlinesParquetManager, and Enhanced Klines Processing Pipeline

This script tests the full compatibility between all components to ensure they work together seamlessly.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_data():
    """Create test klines data for compatibility testing."""
    # Create 1000 minutes of test data
    start_time = datetime.now() - timedelta(minutes=1000)
    timestamps = [start_time + timedelta(minutes=i) for i in range(1000)]
    
    # Generate realistic OHLCV data
    base_price = 3000.0
    prices = []
    volumes = []
    
    for i in range(1000):
        # Random walk with some trend
        price_change = np.random.normal(0, 0.01) * base_price
        base_price += price_change
        base_price = max(base_price, 100.0)  # Prevent negative prices
        
        # Generate OHLC from base price
        high = base_price * (1 + abs(np.random.normal(0, 0.005)))
        low = base_price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = base_price + np.random.normal(0, 0.002) * base_price
        close_price = base_price + np.random.normal(0, 0.002) * base_price
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        prices.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price
        })
        
        # Generate volume
        volume = np.random.uniform(100, 1000)
        volumes.append(volume)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': [p['open'] for p in prices],
        'high': [p['high'] for p in prices],
        'low': [p['low'] for p in prices],
        'close': [p['close'] for p in prices],
        'volume': volumes,
        'exchange': 'binance',
        'symbol': 'ETHUSDT',
        'interval': '1m'
    })
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    return df

async def test_exchange_interface():
    """Test ExchangeInterface functionality."""
    print("🧪 Testing ExchangeInterface...")
    
    try:
        from src.trading.execution.exchange_interface import create_exchange_interface
        
        # Create exchange interface
        config = {
            'exchange_type': 'simulated',
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'testnet': True
        }
        
        exchange = create_exchange_interface(config)
        
        # Test connection
        connected = await exchange.connect()
        assert connected, "Failed to connect to simulated exchange"
        print("✅ ExchangeInterface connection successful")
        
        # Test getting klines
        klines = await exchange.get_klines(
            symbol="ETHUSDT",
            interval="1m",
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now(),
            limit=100
        )
        
        assert len(klines) > 0, "No klines data received"
        print(f"✅ ExchangeInterface klines retrieval successful: {len(klines)} records")
        
        await exchange.disconnect()
        print("✅ ExchangeInterface disconnection successful")
        
        return True
        
    except Exception as e:
        print(f"❌ ExchangeInterface test failed: {e}")
        traceback.print_exc()
        return False

def test_klines_parquet_manager():
    """Test KlinesParquetManager functionality."""
    print("🧪 Testing KlinesParquetManager...")
    
    try:
        from src.utils.kline_parquet import KlinesParquetManager, StorageConfig
        
        # Create test data
        test_data = create_test_data()
        
        # Create storage config
        config = StorageConfig(base_dir="test_data")
        
        # Initialize manager
        manager = KlinesParquetManager(config)
        print("✅ KlinesParquetManager initialization successful")
        
        # Test storing data
        success = manager.store_klines(
            df=test_data,
            symbol="ETHUSDT",
            exchange="binance",
            interval="1m",
            batch_id="test_batch"
        )
        
        assert success, "Failed to store klines data"
        print("✅ KlinesParquetManager storage successful")
        
        # Test loading data
        loaded_data = manager.load_klines(
            symbol="ETHUSDT",
            exchange="binance",
            interval="1m",
            batch_id="test_batch"
        )
        
        assert not loaded_data.empty, "Failed to load klines data"
        assert len(loaded_data) == len(test_data), "Loaded data size mismatch"
        print(f"✅ KlinesParquetManager loading successful: {len(loaded_data)} records")
        
        # Test listing available data
        available_data = manager.list_available_data()
        assert len(available_data) > 0, "No available data found"
        print(f"✅ KlinesParquetManager listing successful: {len(available_data)} files")
        
        return True
        
    except Exception as e:
        print(f"❌ KlinesParquetManager test failed: {e}")
        traceback.print_exc()
        return False

async def test_enhanced_klines_pipeline():
    """Test Enhanced Klines Processing Pipeline functionality."""
    print("🧪 Testing Enhanced Klines Processing Pipeline...")
    
    try:
        from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
            EnhancedKlinesProcessingPipeline,
            PipelineConfig,
            ResamplingConfig
        )
        from src.trading.execution.exchange_interface import create_exchange_interface
        
        # Create test data
        test_data = create_test_data()
        
        # Create exchange interface
        exchange_config = {
            'exchange_type': 'simulated',
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'testnet': True
        }
        exchange = create_exchange_interface(exchange_config)
        await exchange.connect()
        
        # Create pipeline config
        pipeline_config = PipelineConfig(
            data_dir="test_pipeline_data",
            exchange="binance",
            enable_logging=True,
            enable_gap_filling=True,
            enable_resampling=True,
            enable_duplicate_handling=True,
            enable_quality_validation=True,
            batch_compatible=True
        )
        
        # Initialize pipeline
        pipeline = EnhancedKlinesProcessingPipeline(pipeline_config)
        print("✅ Enhanced Klines Processing Pipeline initialization successful")
        
        # Test quality score generation
        quality_score = await pipeline.get_comprehensive_quality_score(
            df=test_data,
            symbol="ETHUSDT",
            interval="1m"
        )
        
        assert quality_score is not None, "Failed to generate quality score"
        print(f"✅ Quality score generation successful: {quality_score.overall_score:.2f}")
        
        # Test data cleaning
        cleaned_data, cleaning_metadata = await pipeline.clean_data_with_quality_utilities(
            df=test_data,
            symbol="ETHUSDT",
            interval="1m"
        )
        
        assert cleaned_data is not None, "Failed to clean data"
        assert len(cleaned_data) > 0, "Cleaned data is empty"
        print(f"✅ Data cleaning successful: {len(cleaned_data)} records")
        
        # Test quality trend analysis
        trend_analysis = await pipeline.analyze_quality_trends(
            df=test_data,
            symbol="ETHUSDT",
            interval="1m",
            window_size=100
        )
        
        assert trend_analysis is not None, "Failed to analyze quality trends"
        print(f"✅ Quality trend analysis successful: {trend_analysis.get('mean_quality', 0):.2f}")
        
        # Test full pipeline processing
        results = await pipeline.process_klines_data(
            symbol="ETHUSDT",
            interval="1m",
            years=1,
            exchange_interface=exchange,
            create_consolidated=True,
            batch_id="compatibility_test"
        )
        
        assert results is not None, "Pipeline processing failed"
        assert results.get('pipeline_success', False), "Pipeline processing was not successful"
        print(f"✅ Full pipeline processing successful: {results.get('final_data_shape', (0, 0))}")
        
        await exchange.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Klines Processing Pipeline test failed: {e}")
        traceback.print_exc()
        return False

async def test_integration():
    """Test full integration between all components."""
    print("🧪 Testing Full Integration...")
    
    try:
        from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
            EnhancedKlinesProcessingPipeline,
            PipelineConfig
        )
        from src.trading.execution.exchange_interface import create_exchange_interface
        from src.utils.kline_parquet import KlinesParquetManager, StorageConfig
        
        # Create exchange interface
        exchange_config = {
            'exchange_type': 'simulated',
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'testnet': True
        }
        exchange = create_exchange_interface(exchange_config)
        await exchange.connect()
        
        # Create pipeline with KlinesParquetManager integration
        pipeline_config = PipelineConfig(
            data_dir="test_integration_data",
            exchange="binance",
            enable_logging=True,
            enable_gap_filling=True,
            enable_resampling=True,
            enable_duplicate_handling=True,
            enable_quality_validation=True,
            batch_compatible=True,
            storage_config=StorageConfig(base_dir="test_integration_data")
        )
        
        pipeline = EnhancedKlinesProcessingPipeline(pipeline_config)
        
        # Test full integration
        results = await pipeline.process_klines_data(
            symbol="ETHUSDT",
            interval="1m",
            years=1,
            exchange_interface=exchange,
            create_consolidated=True,
            batch_id="integration_test"
        )
        
        assert results.get('pipeline_success', False), "Integration test failed"
        assert len(results.get('stored_files', [])) > 0, "No files were stored"
        
        # Verify stored data can be loaded
        manager = pipeline.klines_manager
        loaded_data = manager.load_klines(
            symbol="ETHUSDT",
            exchange="binance",
            interval="1m",
            batch_id="integration_test"
        )
        
        assert not loaded_data.empty, "Failed to load stored data"
        print(f"✅ Full integration test successful: {len(loaded_data)} records stored and loaded")
        
        await exchange.disconnect()
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all compatibility tests."""
    print("🚀 Starting Compatibility Tests...")
    print("=" * 60)
    
    tests = [
        ("ExchangeInterface", test_exchange_interface()),
        ("KlinesParquetManager", test_klines_parquet_manager()),
        ("Enhanced Klines Pipeline", test_enhanced_klines_pipeline()),
        ("Full Integration", test_integration())
    ]
    
    results = []
    
    for test_name, test_coro in tests:
        print(f"\n📋 Running {test_name} test...")
        print("-" * 40)
        
        if asyncio.iscoroutine(test_coro):
            result = await test_coro
        else:
            result = test_coro
            
        results.append((test_name, result))
        
        if result:
            print(f"✅ {test_name} test PASSED")
        else:
            print(f"❌ {test_name} test FAILED")
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:30} | {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All compatibility tests PASSED! Components are fully compatible.")
        return True
    else:
        print("⚠️ Some compatibility tests FAILED. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)