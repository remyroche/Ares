#!/usr/bin/env python3
"""
Create Correct Mock Data for Enhanced Training Manager

This script creates mock data that matches what step1 actually produces
and what the enhanced_training_manager expects for steps 1_5, 2, 3, and 4.

Expected files from step1:
- klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet
- aggtrades_{exchange}_{symbol}_consolidated.parquet
- futures_{exchange}_{symbol}_consolidated.parquet

For ETHUSDT on BINANCE with 1m-30m timeframes.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def create_klines_data(symbol: str, exchange: str, timeframe: str, days: int = 30):
    pass
    pass
    """Create klines data that matches step1 output format."""
    print(f"📊 Creating klines data for {exchange}_{symbol}_{timeframe}")

    # Calculate number of records based on timeframe
    timeframe_minutes = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30
    }

    minutes_per_record = timeframe_minutes.get(timeframe, 1)
    records_per_day = 24 * 60 // minutes_per_record
    total_records = days * records_per_day

    # Generate timestamps
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=days)

    timestamps = pd.date_range(start=start_time, end=end_time, freq=f"{minutes_per_record}min")
    timestamps = timestamps[:total_records]  # Ensure exact number of records

    # Generate realistic ETH price data
    np.random.seed(42)
    base_price = 3000.0
    price = base_price

    klines_data = []

    for i, timestamp in enumerate(timestamps):
    pass
    pass
        # Realistic price movement
        price_change = np.random.normal(0, 0.002)  # 0.2% volatility
        price = max(price * (1 + price_change), 100)

        # Generate OHLCV data
        spread = price * 0.0005  # 0.05% spread
        open_price = price + np.random.uniform(-spread, spread)
        high_price = max(open_price, price + np.random.uniform(0, spread * 2))
        low_price = min(open_price, price - np.random.uniform(0, spread * 2))
        close_price = price + np.random.uniform(-spread, spread)
        volume = np.random.uniform(10, 1000)

        # Convert timestamp to milliseconds
        timestamp_ms = int(timestamp.timestamp() * 1000)

        klines_data.append({
            'timestamp': timestamp_ms,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2),
            'quote_asset_volume': round(volume * close_price, 2),
            'number_of_trades': np.random.randint(1, 100),
            'taker_buy_base_asset_volume': round(volume * 0.6, 2),
            'taker_buy_quote_asset_volume': round(volume * close_price * 0.6, 2),
        })

    df = pd.DataFrame(klines_data)
    print(f"✅ Created {len(df)} klines records")
    return df

def create_aggtrades_data(symbol: str, exchange: str, days: int = 30):
    pass
    pass
    """Create aggtrades data that matches step1 output format."""
    print(f"📊 Creating aggtrades data for {exchange}_{symbol}")

    # Generate timestamps (more frequent than klines)
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=days)

    # Create more frequent timestamps for trades
    timestamps = pd.date_range(start=start_time, end=end_time, freq="30s")

    # Generate realistic trade data
    np.random.seed(42)
    base_price = 3000.0

    aggtrades_data = []

    for timestamp in timestamps:
    pass
    pass
        # Generate multiple trades per timestamp
        num_trades = np.random.randint(1, 5)

        for _ in range(num_trades):
    pass
    pass
            trade_price = base_price + np.random.normal(0, 50)
            quantity = np.random.uniform(0.1, 10.0)

            aggtrades_data.append({
                'timestamp': int(timestamp.timestamp() * 1000),
                'price': round(trade_price, 2),
                'quantity': round(quantity, 4),
                'first_trade_id': np.random.randint(1000000, 9999999),
                'last_trade_id': np.random.randint(1000000, 9999999),
                'trade_time': int(timestamp.timestamp() * 1000),
                'is_buyer_maker': np.random.choice([True, False]),
            })

    df = pd.DataFrame(aggtrades_data)
    print(f"✅ Created {len(df)} aggtrades records")
    return df

def create_futures_data(symbol: str, exchange: str, days: int = 30):
    pass
    pass
    """Create futures data that matches step1 output format (8h funding rate data)."""
    print(f"📊 Creating futures data for {exchange}_{symbol}")

    # Generate 8-hour intervals for funding rate data
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=days)

    # Create 8-hour intervals
    timestamps = pd.date_range(start=start_time, end=end_time, freq="8H")

    np.random.seed(42)
    base_price = 3000.0

    futures_data = []

    for timestamp in timestamps:
    pass
    pass
        # Generate futures data
        mark_price = base_price + np.random.normal(0, 100)
        index_price = mark_price + np.random.normal(0, 10)
        funding_rate = np.random.uniform(-0.01, 0.01)  # -1% to +1%
        next_funding_time = timestamp + timedelta(hours=8)

        futures_data.append({
            'timestamp': int(timestamp.timestamp() * 1000),
            'symbol': symbol,
            'mark_price': round(mark_price, 2),
            'index_price': round(index_price, 2),
            'funding_rate': round(funding_rate, 6),
            'next_funding_time': int(next_funding_time.timestamp() * 1000),
        })

    df = pd.DataFrame(futures_data)
    print(f"✅ Created {len(df)} futures records")
    return df

def create_unified_data(symbol: str, exchange: str, timeframe: str, days: int = 30):
    pass
    pass
    """Create unified data that step1_5 produces."""
    print(f"📊 Creating unified data for {exchange}_{symbol}_{timeframe}")

    # Load the klines data to create unified format
    klines_file = f"data_cache/klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"

    if os.path.exists(klines_file):
    pass
    pass
        klines_df = pd.read_parquet(klines_file)

        # Create unified format
        unified_data = klines_df.copy()

        # Add additional columns that step1_5 creates
        unified_data['exchange'] = exchange
        unified_data['symbol'] = symbol
        unified_data['timeframe'] = timeframe

        # Convert timestamp to datetime
        unified_data['datetime'] = pd.to_datetime(unified_data['timestamp'], unit='ms', utc=True)

        # Add some derived features
        unified_data['price_change'] = unified_data['close'].pct_change()
        unified_data['volume_ma'] = unified_data['volume'].rolling(window=20).mean()

        print(f"✅ Created unified dataset with {len(unified_data)} records")
        return unified_data
    else:
        print(f"❌ Klines file not found: {klines_file}")
        return None

def create_features_data(symbol: str, exchange: str, timeframe: str, days: int = 30):
    pass
    pass
    """Create features data that step2 produces."""
    print(f"📊 Creating features data for {exchange}_{symbol}_{timeframe}")

    # Load unified data
    unified_file = f"data_cache/unified_{exchange}_{symbol}_{timeframe}.parquet"

    if os.path.exists(unified_file):
    pass
    pass
        unified_df = pd.read_parquet(unified_file)

        # Create features dataset
        features_data = unified_df.copy()

        # Add technical indicators (simplified)
        features_data['sma_20'] = features_data['close'].rolling(window=20).mean()
        features_data['sma_50'] = features_data['close'].rolling(window=50).mean()
        features_data['rsi'] = calculate_rsi(features_data['close'])
        features_data['volatility'] = features_data['close'].rolling(window=20).std()

        # Add some engineered features
        features_data['price_momentum'] = features_data['close'].pct_change(5)
        features_data['volume_momentum'] = features_data['volume'].pct_change(5)
        features_data['high_low_ratio'] = features_data['high'] / features_data['low']

        # Remove NaN values
        features_data = features_data.dropna()

        print(f"✅ Created features dataset with {len(features_data)} records")
        return features_data
    else:
        print(f"❌ Unified file not found: {unified_file}")
        return None

def calculate_rsi(prices, period=14):
    pass
    pass
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def create_all_mock_data(symbol: str = "ETHUSDT", exchange: str = "BINANCE", days: int = 30):
    pass
    pass
    """Create all required mock data for the pipeline."""
    print("🚀 Creating Complete Mock Data Set")
    print("=" * 60)

    # Create data_cache directory
    os.makedirs("data_cache", exist_ok=True)
    os.makedirs("data/training", exist_ok=True)

    # Create data for different timeframes
    timeframes = ["1m", "3m", "5m", "15m", "30m"]

    for timeframe in timeframes:
    pass
    pass
        print(f"\\\n📊 Processing timeframe: {timeframe}")
        print("-" * 40)

        # 1. Create klines data (step1 output)
        klines_df = create_klines_data(symbol, exchange, timeframe, days)
        klines_file = f"data_cache/klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet"
        klines_df.to_parquet(klines_file, index=False)
        print(f"💾 Saved klines: {klines_file}")

        # 2. Create aggtrades data (step1 output)
        if timeframe == "1m":  # Only create once for aggtrades
            aggtrades_df = create_aggtrades_data(symbol, exchange, days)
            aggtrades_file = f"data_cache/aggtrades_{exchange}_{symbol}_consolidated.parquet"
            aggtrades_df.to_parquet(aggtrades_file, index=False)
            print(f"💾 Saved aggtrades: {aggtrades_file}")

        # 3. Create futures data (step1 output)
        if timeframe == "1m":  # Only create once for futures
            futures_df = create_futures_data(symbol, exchange, days)
            futures_file = f"data_cache/futures_{exchange}_{symbol}_consolidated.parquet"
            futures_df.to_parquet(futures_file, index=False)
            print(f"💾 Saved futures: {futures_file}")

        # 4. Create unified data (step1_5 output)
        unified_df = create_unified_data(symbol, exchange, timeframe, days)
        if unified_df is not None:
    pass
    pass
            unified_file = f"data_cache/unified_{exchange}_{symbol}_{timeframe}.parquet"
            unified_df.to_parquet(unified_file, index=False)
            print(f"💾 Saved unified: {unified_file}")

            # Create config file
            config_data = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'created_at': datetime.now().isoformat(),
                'data_points': len(unified_df),
                'start_date': unified_df['datetime'].min().isoformat(),
                'end_date': unified_df['datetime'].max().isoformat()
            }

            import json
            config_file = f"data_cache/unified_{exchange}_{symbol}_{timeframe}_config.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"💾 Saved config: {config_file}")

        # 5. Create features data (step2 output)
        features_df = create_features_data(symbol, exchange, timeframe, days)
        if features_df is not None:
    pass
    pass
            # Split into train/val/test
            total_rows = len(features_df)
            train_size = int(total_rows * 0.7)
            val_size = int(total_rows * 0.15)

            train_df = features_df[:train_size]
            val_df = features_df[train_size:train_size + val_size]
            test_df = features_df[train_size + val_size:]

            # Save splits
            train_file = f"data/training/features_{exchange}_{symbol}_{timeframe}_train.parquet"
            val_file = f"data/training/features_{exchange}_{symbol}_{timeframe}_val.parquet"
            test_file = f"data/training/features_{exchange}_{symbol}_{timeframe}_test.parquet"

            train_df.to_parquet(train_file, index=False)
            val_df.to_parquet(val_file, index=False)
            test_df.to_parquet(test_file, index=False)

            print(f"💾 Saved features train: {train_file} ({len(train_df)} records)")
            print(f"💾 Saved features val: {val_file} ({len(val_df)} records)")
            print(f"💾 Saved features test: {test_file} ({len(test_df)} records)")

    print("\\\n" + "=" * 60)
    print("✅ All mock data created successfully!")
    print("=" * 60)

    # List all created files
    print("\\\n📁 Created files:")
    data_cache_files = list(Path("data_cache").glob("*.parquet"))
    training_files = list(Path("data/training").glob("*.parquet"))

    for file in data_cache_files:
    pass
    pass
        size = file.stat().st_size
        print(f"   📊 {file}: {size:,} bytes")

    for file in training_files:
    pass
    pass
        size = file.stat().st_size
        print(f"   📊 {file}: {size:,} bytes")

def main():
    pass
    pass
    """Main function to create mock data."""
    print("🎯 Creating Correct Mock Data for Enhanced Training Manager")
    print("=" * 80)
    print("This will create mock data that matches:")
    print("- Step1 outputs: klines, aggtrades, futures")
    print("- Step1_5 outputs: unified datasets")
    print("- Step2 outputs: feature datasets")
    print("- For ETHUSDT on BINANCE with 1m-30m timeframes")
    print("=" * 80)

    create_all_mock_data("ETHUSDT", "BINANCE", days=30)

    print("\\\n🎉 Mock data creation completed!")
    print("The enhanced_training_manager can now use this data for steps 1_5, 2, 3, and 4.")

if __name__ == "__main__":
    pass
    pass
    main()