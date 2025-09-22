#!/usr/bin/env python3
"""
Create Sample Data for Testing

This script creates sample market data files to test the training pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_sample_klines_data(symbol: str, timeframe: str, days: int = 30):
    """Create sample klines data."""
    print(f"📊 Creating sample data for {symbol} {timeframe} ({days} days)")
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Calculate number of candles based on timeframe
    if timeframe == "1m":
        freq = "1T"
        candles_per_day = 1440
    elif timeframe == "5m":
        freq = "5T"
        candles_per_day = 288
    elif timeframe == "15m":
        freq = "15T"
        candles_per_day = 96
    elif timeframe == "1h":
        freq = "1H"
        candles_per_day = 24
    else:
        freq = "1H"
        candles_per_day = 24
    
    # Generate timestamp index
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_candles = len(timestamps)
    
    print(f"   📅 Date range: {start_date} to {end_date}")
    print(f"   📊 Generated {n_candles} candles")
    
    # Generate realistic price data (starting around 2500 for ETHUSDT)
    np.random.seed(42)  # For reproducible data
    base_price = 2500.0
    
    # Generate price movements with some trend and volatility
    price_changes = np.random.normal(0, 0.02, n_candles)  # 2% volatility
    price_trend = np.linspace(0, 0.1, n_candles)  # 10% upward trend over period
    
    # Calculate prices
    cumulative_changes = np.cumsum(price_changes + price_trend / n_candles)
    close_prices = base_price * (1 + cumulative_changes)
    
    # Generate OHLC data
    high_noise = np.random.uniform(0.001, 0.01, n_candles)  # 0.1-1% above close
    low_noise = np.random.uniform(-0.01, -0.001, n_candles)  # 0.1-1% below close
    open_shift = np.random.uniform(-0.005, 0.005, n_candles)  # ±0.5% from previous close
    
    # Create OHLC arrays
    open_prices = np.zeros(n_candles)
    high_prices = np.zeros(n_candles)
    low_prices = np.zeros(n_candles)
    
    for i in range(n_candles):
        if i == 0:
            open_prices[i] = base_price
        else:
            open_prices[i] = close_prices[i-1] * (1 + open_shift[i])
        
        high_prices[i] = max(open_prices[i], close_prices[i]) * (1 + high_noise[i])
        low_prices[i] = min(open_prices[i], close_prices[i]) * (1 + low_noise[i])
    
    # Generate volume data
    base_volume = 1000000  # 1M base volume
    volume_noise = np.random.uniform(0.5, 2.0, n_candles)
    volumes = base_volume * volume_noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes,
        'quote_volume': volumes * close_prices,
        'trades': np.random.randint(100, 1000, n_candles),
        'taker_buy_volume': volumes * np.random.uniform(0.4, 0.6, n_candles),
        'taker_buy_quote_volume': volumes * close_prices * np.random.uniform(0.4, 0.6, n_candles)
    })
    
    # Set timestamp as index
    data.set_index('timestamp', inplace=True)
    
    print(f"   ✅ Generated data with columns: {list(data.columns)}")
    print(f"   📊 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"   📊 Volume range: {data['volume'].min():.0f} - {data['volume'].max():.0f}")
    
    return data

def save_sample_data():
    """Save sample data files."""
    print("🚀 Creating sample data files for testing...")
    
    # Create data for ETHUSDT 15m
    data_15m = create_sample_klines_data("ETHUSDT", "15m", days=30)
    
    # Save to parquet files
    data_dir = Path("historical_data/binance/ETHUSDT/15m")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as processed data
    processed_file = data_dir / "processed_data.parquet"
    data_15m.to_parquet(processed_file)
    print(f"   ✅ Saved processed data: {processed_file}")
    
    # Save as raw data (same content for testing)
    raw_file = data_dir / "raw_data.parquet"
    data_15m.to_parquet(raw_file)
    print(f"   ✅ Saved raw data: {raw_file}")
    
    # Create additional timeframe data
    print("\n📊 Creating additional timeframe data...")
    
    # Create 1h data (resample from 15m)
    data_1h = data_15m.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trades': 'sum',
        'taker_buy_volume': 'sum',
        'taker_buy_quote_volume': 'sum'
    }).dropna()
    
    data_dir_1h = Path("historical_data/binance/ETHUSDT/1h")
    data_dir_1h.mkdir(parents=True, exist_ok=True)
    
    processed_file_1h = data_dir_1h / "processed_data.parquet"
    data_1h.to_parquet(processed_file_1h)
    print(f"   ✅ Saved 1h processed data: {processed_file_1h}")
    
    # Create some features for testing
    print("\n🔧 Adding technical indicators...")
    
    # Add simple moving averages
    data_15m['sma_20'] = data_15m['close'].rolling(20).mean()
    data_15m['sma_50'] = data_15m['close'].rolling(50).mean()
    
    # Add RSI (simple approximation)
    delta = data_15m['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data_15m['rsi'] = 100 - (100 / (1 + rs))
    
    # Add returns
    data_15m['returns'] = data_15m['close'].pct_change()
    data_15m['log_returns'] = np.log(data_15m['close']).diff()
    
    # Add volatility
    data_15m['volatility'] = data_15m['returns'].rolling(20).std()
    
    # Save enhanced data
    enhanced_file = data_dir / "enhanced_data.parquet"
    data_15m.to_parquet(enhanced_file)
    print(f"   ✅ Saved enhanced data with features: {enhanced_file}")
    
    print(f"\n🎉 Sample data creation completed!")
    print(f"   📊 15m data: {len(data_15m)} rows")
    print(f"   📊 1h data: {len(data_1h)} rows")
    print(f"   📂 Data directory: historical_data/")
    
    return data_15m, data_1h

if __name__ == "__main__":
    try:
        data_15m, data_1h = save_sample_data()
        print("\n✅ Sample data creation successful!")
    except Exception as e:
        print(f"\n❌ Sample data creation failed: {e}")
        import traceback
        traceback.print_exc()