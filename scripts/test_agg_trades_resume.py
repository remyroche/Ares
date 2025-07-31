#!/usr/bin/env python3
"""
Test script to check aggregated trades resume functionality.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.data_downloader import download_agg_trades_data
from src.utils.logger import system_logger

async def test_agg_trades_resume():
    """Test the aggregated trades resume functionality."""
    logger = system_logger
    
    logger.info("Testing aggregated trades resume functionality...")
    
    # Create a mock client
    class MockClient:
        async def get_historical_agg_trades_ccxt(self, symbol, start_time_ms, end_time_ms):
            logger.info(f"Mock: Would fetch agg trades for {symbol} from {datetime.fromtimestamp(start_time_ms / 1000)} to {datetime.fromtimestamp(end_time_ms / 1000)}")
            return []
    
    client = MockClient()
    
    # Test with a short lookback period
    success = await download_agg_trades_data(client, "BINANCE", "ETHUSDT", 0.1)
    
    logger.info(f"Test completed. Success: {success}")

if __name__ == "__main__":
    asyncio.run(test_agg_trades_resume()) 