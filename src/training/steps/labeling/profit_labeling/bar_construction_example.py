"""
Example usage of bar construction methods for crypto trading.

This file demonstrates how to use the various bar construction methods
implemented in bar_construction.py for crypto data analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bar_construction import (
    BarConstructor, BarConstructionConfig, BarType,
    create_crypto_dollar_bar_constructor,
    create_crypto_volume_bar_constructor,
    create_crypto_range_bar_constructor
)


def create_sample_crypto_data(n_points: int = 1000) -> pd.DataFrame:
    """Create sample crypto data for testing bar construction methods."""
    # Generate sample timestamps
    start_time = datetime.now() - timedelta(hours=24)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_points)]
    
    # Generate sample price data (random walk)
    np.random.seed(42)  # For reproducible results
    base_price = 50000.0  # BTC-like price
    price_changes = np.random.normal(0, 0.001, n_points)  # 0.1% volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        # Add some noise to create realistic OHLC
        noise_factor = 0.0005  # 0.05% noise
        high = price * (1 + abs(np.random.normal(0, noise_factor)))
        low = price * (1 - abs(np.random.normal(0, noise_factor)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        
        # Generate volume (higher volume during price movements)
        price_change = abs(price - open_price) / open_price if i > 0 else 0
        base_volume = 1000
        volume = base_volume * (1 + price_change * 10) * np.random.uniform(0.5, 2.0)
        
        data.append({
            'open': open_price,
            'high': max(open_price, high, close_price),
            'low': min(open_price, low, close_price),
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=timestamps)
    return df


def demonstrate_volume_bars():
    """Demonstrate volume bar construction."""
    print("=== Volume Bar Construction ===")
    
    # Create sample data
    data = create_sample_crypto_data(500)
    print(f"Original data: {len(data)} points")
    
    # Create volume bar constructor
    constructor = create_crypto_volume_bar_constructor(bar_size=5000.0)
    
    # Construct volume bars
    volume_bars = constructor.construct_bars(data)
    print(f"Volume bars: {len(volume_bars)} bars")
    
    # Get statistics
    stats = constructor.get_bar_statistics(volume_bars)
    print(f"Volume statistics: {stats['volume_stats']}")
    
    return volume_bars


def demonstrate_dollar_bars():
    """Demonstrate dollar bar construction for crypto."""
    print("\n=== Dollar Bar Construction (Crypto) ===")
    
    # Create sample data
    data = create_sample_crypto_data(500)
    print(f"Original data: {len(data)} points")
    
    # Create dollar bar constructor
    constructor = create_crypto_dollar_bar_constructor(
        bar_size=50000.0,  # $50k threshold
        quote_currency='USDT'
    )
    
    # Construct dollar bars
    dollar_bars = constructor.construct_bars(data, quote_currency='USDT')
    print(f"Dollar bars: {len(dollar_bars)} bars")
    
    # Get statistics
    stats = constructor.get_bar_statistics(dollar_bars)
    print(f"Volume statistics: {stats['volume_stats']}")
    
    return dollar_bars


def demonstrate_tick_bars():
    """Demonstrate tick bar construction."""
    print("\n=== Tick Bar Construction ===")
    
    # Create sample data
    data = create_sample_crypto_data(500)
    print(f"Original data: {len(data)} points")
    
    # Create tick bar constructor
    config = BarConstructionConfig(
        bar_type=BarType.TICK,
        bar_size=50.0,  # 50 ticks per bar
        min_bars_required=5
    )
    constructor = BarConstructor(config)
    
    # Construct tick bars
    tick_bars = constructor.construct_bars(
        data, 
        tick_column='close',
        price_change_threshold=0.0001  # 0.01% minimum change
    )
    print(f"Tick bars: {len(tick_bars)} bars")
    
    return tick_bars


def demonstrate_range_bars():
    """Demonstrate range bar construction."""
    print("\n=== Range Bar Construction ===")
    
    # Create sample data
    data = create_sample_crypto_data(500)
    print(f"Original data: {len(data)} points")
    
    # Create range bar constructor
    constructor = create_crypto_range_bar_constructor(
        bar_size=0.005,  # 0.5% range
        range_type='hl',
        use_atr=True
    )
    
    # Construct range bars
    range_bars = constructor.construct_bars(
        data,
        range_type='hl',
        use_atr=True,
        atr_period=14
    )
    print(f"Range bars: {len(range_bars)} bars")
    
    # Get statistics
    stats = constructor.get_bar_statistics(range_bars)
    print(f"Price statistics: {stats['price_stats']}")
    
    return range_bars


def demonstrate_time_bars():
    """Demonstrate time bar construction."""
    print("\n=== Time Bar Construction ===")
    
    # Create sample data
    data = create_sample_crypto_data(500)
    print(f"Original data: {len(data)} points")
    
    # Create time bar constructor
    config = BarConstructionConfig(
        bar_type=BarType.TIME,
        bar_size=1.0,
        min_bars_required=5
    )
    constructor = BarConstructor(config)
    
    # Construct time bars (5-minute bars)
    time_bars = constructor.construct_bars(
        data,
        timeframe='5T',  # 5-minute bars
        resample_method='ohlc'
    )
    print(f"Time bars: {len(time_bars)} bars")
    
    return time_bars


if __name__ == "__main__":
    print("Crypto Bar Construction Examples")
    print("=" * 50)
    
    # Run all demonstrations
    volume_bars = demonstrate_volume_bars()
    dollar_bars = demonstrate_dollar_bars()
    tick_bars = demonstrate_tick_bars()
    range_bars = demonstrate_range_bars()
    time_bars = demonstrate_time_bars()
    
    print("\n=== Summary ===")
    print(f"Volume bars: {len(volume_bars)} bars")
    print(f"Dollar bars: {len(dollar_bars)} bars")
    print(f"Tick bars: {len(tick_bars)} bars")
    print(f"Range bars: {len(range_bars)} bars")
    print(f"Time bars: {len(time_bars)} bars")
    
    print("\nBar construction methods successfully implemented!")