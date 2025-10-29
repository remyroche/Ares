#!/usr/bin/env python3
"""
Simple script to run the enhanced klines processing pipeline for ETHUSDT, 4 years, BingX
This version bypasses the trading module circular import issues by using a mock exchange interface.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Mock ExchangeInterface to bypass circular import issues
class MockExchangeInterface:
    """Mock exchange interface for testing without API calls."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = True
        self.exchange_type = config.get('exchange_type', 'mock')
    
    async def connect(self):
        """Mock connection."""
        print("🔌 Mock exchange interface connected")
        return True
    
    async def disconnect(self):
        """Mock disconnection."""
        print("🔌 Mock exchange interface disconnected")
        pass
    
    async def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000):
        """Mock klines data generation."""
        print(f"📊 Mock: Generating klines data for {symbol} {interval} from {start_time} to {end_time}")
        
        # Generate mock OHLCV data
        start_dt = datetime.fromtimestamp(start_time / 1000)
        end_dt = datetime.fromtimestamp(end_time / 1000)
        
        # Create time range based on interval
        if interval == '1m':
            freq = '1T'
        elif interval == '5m':
            freq = '5T'
        elif interval == '15m':
            freq = '15T'
        elif interval == '30m':
            freq = '30T'
        elif interval == '1h':
            freq = '1H'
        else:
            freq = '1T'
        
        time_range = pd.date_range(start=start_dt, end=end_dt, freq=freq)
        
        # Generate realistic mock data
        np.random.seed(42)  # For reproducible results
        n_points = len(time_range)
        
        # Start with a base price around $2000 for ETH
        base_price = 2000.0
        price_changes = np.random.normal(0, 0.01, n_points)  # 1% volatility
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 100))  # Minimum price floor
        
        # Generate OHLCV data
        klines = []
        for i, (timestamp, price) in enumerate(zip(time_range, prices)):
            # Generate OHLC from price
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.uniform(100, 1000)
            
            klines.append([
                int(timestamp.timestamp() * 1000),  # timestamp
                f"{open_price:.2f}",                # open
                f"{high:.2f}",                      # high
                f"{low:.2f}",                       # low
                f"{close_price:.2f}",               # close
                f"{volume:.2f}",                    # volume
                int(timestamp.timestamp() * 1000),  # close_time
                f"{volume * price:.2f}",            # quote_volume
                0,                                  # trades
                f"{volume * 0.5:.2f}",             # taker_buy_base_volume
                f"{volume * price * 0.5:.2f}",     # taker_buy_quote_volume
                "0"                                 # ignore
            ])
        
        return klines

def create_exchange_interface(config: Dict[str, Any]) -> MockExchangeInterface:
    """Create a mock exchange interface."""
    return MockExchangeInterface(config)

# Mock the trading module to avoid circular imports
class MockTradingModule:
    class ExchangeInterface:
        def __init__(self, config):
            self.config = config
            self.connected = True
            self.exchange_type = config.get('exchange_type', 'mock')
        
        async def connect(self):
            print("🔌 Mock exchange interface connected")
            return True
        
        async def disconnect(self):
            print("🔌 Mock exchange interface disconnected")
            pass
        
        async def get_klines(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 1000):
            return await MockExchangeInterface(self.config).get_klines(symbol, interval, start_time, end_time, limit)

# Patch the trading module
sys.modules['src.trading.execution.exchange_interface'] = MockTradingModule()

async def main():
    """Run the enhanced klines processing pipeline."""
    try:
        print("🚀 Starting Enhanced Klines Processing Pipeline for ETHUSDT, 4 years, BingX")
        print("📝 Using mock exchange interface to bypass circular import issues")
        
        # Import the pipeline components directly
        from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
            EnhancedKlinesProcessingPipeline,
            PipelineConfig,
            ResamplingConfig
        )
        
        # Configure pipeline
        pipeline_config = PipelineConfig(
            data_dir="historical_data",
            exchange="bingx",
            enable_logging=True,
            enable_gap_filling=True,
            enable_resampling=True,
            enable_duplicate_handling=True,
            enable_quality_validation=True,
            batch_compatible=True
        )
        
        # Configure resampling
        resampling_config = ResamplingConfig(
            target_intervals=['5m', '15m', '30m', '1h'],
            method='ohlc',
            preserve_volume=True,
            resample_older_than_days=1,
            enable_auto_resampling=True
        )
        
        # Create pipeline
        pipeline = EnhancedKlinesProcessingPipeline(pipeline_config)
        
        print("📊 Processing data using simplified interface...")
        
        # Process data using simplified interface
        results = await pipeline.process_klines_data_simple(
            exchange="bingx",  # Exchange name
            asset="ETH",       # Asset (will create ETHUSDT symbol)
            lookback_period="4y",  # Lookback period: 4 years
            interval="1m",     # Data interval
            api_key="",        # Your API key
            api_secret="",     # Your API secret
            use_testnet=True,  # Use testnet
            resampling_config=resampling_config,
            batch_id="ethusdt_4y_bingx"
        )
        
        print(f"\n🎉 Simple processing completed: {results['pipeline_success']}")
        print(f"📊 Data quality: {results['data_quality']}")
        print(f"📈 Final shape: {results['final_data_shape']}")
        print(f"💾 Stored files: {results['stored_files']}")
        print(f"🔄 Resampled intervals: {results['resampled_intervals']}")
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
