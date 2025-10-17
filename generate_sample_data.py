#!/usr/bin/env python3
"""
Generate sample market data for testing the tactician labeler.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_sample_data(symbol="ETHUSDT", timeframe="15m", days=30):
    """Generate sample OHLCV data for testing."""
    
    # Calculate number of periods based on timeframe
    if timeframe == "15m":
        periods_per_day = 24 * 4  # 15 minutes = 4 periods per hour
    elif timeframe == "1h":
        periods_per_day = 24
    elif timeframe == "1m":
        periods_per_day = 24 * 60
    else:
        periods_per_day = 24 * 4  # Default to 15m
    
    total_periods = days * periods_per_day
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=days)
    timestamps = [start_date + timedelta(minutes=15*i) for i in range(total_periods)]
    
    # Generate price data with some realistic patterns
    np.random.seed(42)  # For reproducible results
    
    # Starting price
    base_price = 3000.0
    
    # Generate price movements
    returns = np.random.normal(0, 0.02, total_periods)  # 2% volatility
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.01))  # 1% intraday volatility
        
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = close * (1 + np.random.normal(0, 0.005))  # Small gap
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume (higher volume during volatile periods)
        volume = np.random.exponential(1000) * (1 + volatility * 10)
        
        data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': int(volume)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def save_sample_data(symbol="ETHUSDT", timeframe="15m", days=30):
    """Generate and save sample data to the expected directory structure."""
    
    # Create directory structure
    data_dir = f"historical_data/binance/{symbol.lower()}/processed"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate data
    df = generate_sample_data(symbol, timeframe, days)
    
    # Save as parquet
    filename = f"{symbol.lower()}_{timeframe}_sample_data.parquet"
    filepath = os.path.join(data_dir, filename)
    df.to_parquet(filepath)
    
    print(f"✅ Generated sample data: {filepath}")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    return filepath

if __name__ == "__main__":
    # Generate sample data for ETHUSDT 15m
    save_sample_data("ETHUSDT", "15m", 30)