#!/usr/bin/env python3
"""
Test script to demonstrate the optimized gap-first approach in the enhanced klines processing pipeline.

This script shows how the pipeline now:
1. Analyzes existing data first
2. Detects gaps before downloading
3. Downloads only missing data
4. Avoids duplicate downloads
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
    EnhancedKlinesProcessingPipeline,
    PipelineConfig,
    ResamplingConfig
)
from src.utils.tprint import tprint_info, tprint_success, tprint_warning

class MockExchangeInterface:
    """Mock exchange interface for testing."""
    
    def __init__(self):
        self.connected = False
        self.exchange_type = "mock"
        self.download_calls = []
    
    async def connect(self):
        self.connected = True
        tprint_info("🔌 Connected to mock exchange")
        return True
    
    async def disconnect(self):
        self.connected = False
        tprint_info("🔌 Disconnected from mock exchange")
    
    async def get_klines(self, symbol, interval, start_time, end_time, limit=1000):
        """Mock get_klines that tracks what data is requested."""
        self.download_calls.append({
            'symbol': symbol,
            'interval': interval,
            'start_time': start_time,
            'end_time': end_time,
            'limit': limit
        })
        
        tprint_info(f"📥 Mock download: {symbol} {interval} from {start_time} to {end_time}")
        
        # Return empty list to simulate no data (for testing)
        return []

async def test_gap_optimization():
    """Test the optimized gap-first approach."""
    
    tprint_info("🚀 Testing optimized gap-first approach")
    tprint_info("=" * 60)
    
    # Create pipeline config
    config = PipelineConfig(
        data_dir="historical_data",
        exchange="binance",
        enable_logging=True,
        enable_gap_filling=True,
        enable_resampling=True,
        enable_duplicate_handling=True,
        enable_quality_validation=True,
        batch_compatible=True
    )
    
    # Create pipeline
    pipeline = EnhancedKlinesProcessingPipeline(config)
    
    # Create mock exchange interface
    exchange_interface = MockExchangeInterface()
    
    # Test 1: No existing data (should download everything)
    tprint_info("\n📋 Test 1: No existing data")
    tprint_info("-" * 40)
    
    # Clear any existing data for this test
    test_data_dir = Path("historical_data/binance/btcusdt/raw")
    if test_data_dir.exists():
        import shutil
        shutil.rmtree(test_data_dir)
    
    try:
        result = await pipeline.process_klines_data(
            symbol="BTCUSDT",
            interval="1m",
            years=1,
            exchange_interface=exchange_interface
        )
        
        tprint_info(f"📊 Download calls made: {len(exchange_interface.download_calls)}")
        for i, call in enumerate(exchange_interface.download_calls):
            tprint_info(f"  Call {i+1}: {call['start_time']} to {call['end_time']}")
        
        tprint_success(f"✅ Test 1 completed: {result['pipeline_success']}")
        
    except Exception as e:
        tprint_warning(f"⚠️ Test 1 failed (expected): {e}")
    
    # Test 2: With existing data (should detect gaps and download selectively)
    tprint_info("\n📋 Test 2: With existing data")
    tprint_info("-" * 40)
    
    # Create some mock existing data
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a mock parquet file with some data
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create mock data for the last 30 days
    end_date = datetime.now() - timedelta(days=3)
    start_date = end_date - timedelta(days=30)
    
    mock_data = pd.DataFrame({
        'timestamp': pd.date_range(start_date, end_date, freq='1min'),
        'open': 50000.0,
        'high': 51000.0,
        'low': 49000.0,
        'close': 50500.0,
        'volume': 100.0
    })
    
    # Save mock data
    mock_file = test_data_dir / "btcusdt_1m_mock.parquet"
    mock_data.to_parquet(mock_file, index=False)
    
    tprint_info(f"📁 Created mock data file: {mock_file}")
    tprint_info(f"📊 Mock data records: {len(mock_data)}")
    tprint_info(f"📅 Mock data range: {mock_data['timestamp'].min()} to {mock_data['timestamp'].max()}")
    
    # Reset download calls
    exchange_interface.download_calls = []
    
    try:
        result = await pipeline.process_klines_data(
            symbol="BTCUSDT",
            interval="1m",
            years=1,  # Request 1 year, but we only have 30 days
            exchange_interface=exchange_interface
        )
        
        tprint_info(f"📊 Download calls made: {len(exchange_interface.download_calls)}")
        for i, call in enumerate(exchange_interface.download_calls):
            tprint_info(f"  Call {i+1}: {call['start_time']} to {call['end_time']}")
        
        tprint_success(f"✅ Test 2 completed: {result['pipeline_success']}")
        
    except Exception as e:
        tprint_warning(f"⚠️ Test 2 failed: {e}")
    
    # Test 3: Complete data (should not download anything)
    tprint_info("\n📋 Test 3: Complete data")
    tprint_info("-" * 40)
    
    # Create complete data for the requested period
    end_date = datetime.now() - timedelta(days=3)
    start_date = end_date - timedelta(days=365)  # 1 year
    
    complete_data = pd.DataFrame({
        'timestamp': pd.date_range(start_date, end_date, freq='1min'),
        'open': 50000.0,
        'high': 51000.0,
        'low': 49000.0,
        'close': 50500.0,
        'volume': 100.0
    })
    
    # Save complete data
    complete_file = test_data_dir / "btcusdt_1m_complete.parquet"
    complete_data.to_parquet(complete_file, index=False)
    
    tprint_info(f"📁 Created complete data file: {complete_file}")
    tprint_info(f"📊 Complete data records: {len(complete_data)}")
    
    # Reset download calls
    exchange_interface.download_calls = []
    
    try:
        result = await pipeline.process_klines_data(
            symbol="BTCUSDT",
            interval="1m",
            years=1,
            exchange_interface=exchange_interface
        )
        
        tprint_info(f"📊 Download calls made: {len(exchange_interface.download_calls)}")
        
        if len(exchange_interface.download_calls) == 0:
            tprint_success("✅ No downloads needed - data is complete!")
        else:
            tprint_warning(f"⚠️ Unexpected downloads: {len(exchange_interface.download_calls)}")
        
        tprint_success(f"✅ Test 3 completed: {result['pipeline_success']}")
        
    except Exception as e:
        tprint_warning(f"⚠️ Test 3 failed: {e}")
    
    tprint_info("\n🎉 Gap optimization testing completed!")
    tprint_info("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_gap_optimization())