#!/usr/bin/env python3
"""
Test script for adaptive thresholding in orthogonal label generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from training.steps.labeling.orthogonal_label_generation import (
    orthogonal_label_generation,
    EntropyEvents,
    MicrostructureEvents,
    ATRShockEvents,
    VWAPReversionEvents,
    KalmanRegimeEvents
)

def create_sample_data(days=30):
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=days*24*4, freq='15min')  # 15-min bars
    
    # Simulate price with some trends and volatility
    np.random.seed(42)
    returns = np.random.normal(0, 0.002, len(dates))
    
    # Add some regime changes
    regime_changes = np.random.choice([0, 1], size=len(dates), p=[0.7, 0.3])
    volatility = np.where(regime_changes == 1, 0.004, 0.001)
    returns *= volatility
    
    price = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLC
    high = price * (1 + np.abs(np.random.normal(0, 0.001, len(dates))))
    low = price * (1 - np.abs(np.random.normal(0, 0.001, len(dates))))
    open_price = np.roll(price, 1)
    open_price[0] = price[0]
    
    # Volume
    volume = np.random.lognormal(10, 1, len(dates))
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': price,
        'volume': volume
    }, index=dates)
    
    # Add volatility column
    df['volatility_1d'] = df['close'].pct_change().rolling(96).std()  # 24h volatility
    
    return df

def test_adaptive_thresholds():
    """Test adaptive vs non-adaptive thresholding."""
    print("=== Testing Adaptive Thresholding ===")
    
    # Create sample data
    df = create_sample_data(days=30)
    print(f"Created sample data: {len(df)} bars over {(df.index[-1] - df.index[0]).days} days")
    
    # Test individual generators
    generators_to_test = [
        ('Entropy', EntropyEvents(), {'window': 24, 'z_thresh': 2.0}),
        ('ATRShock', ATRShockEvents(), {'lookback': 14, 'long_window': 50, 'z': 2.0}),
        ('Microstructure', MicrostructureEvents(), {'window': 20, 'z': 2.0}),
        ('VWAPReversion', VWAPReversionEvents(), {'lookback': 50, 'z': 2.5}),
        ('KalmanRegime', KalmanRegimeEvents(), {'Q': 1e-4, 'R': 0.01, 'z': 2.0}),
    ]
    
    target_signals_per_day = 7.5
    
    print(f"\nTarget: {target_signals_per_day} signals per day")
    print("-" * 60)
    
    for name, gen, params in generators_to_test:
        # Standard generation
        try:
            if name in ['ATRShock', 'Microstructure', 'VWAPReversion', 'KalmanRegime']:
                events_std = gen.generate(df, **params)
            else:
                events_std = gen.generate(df['close'], **params)
            
            duration_days = (events_std[-1] - events_std[0]).days if len(events_std) > 1 else 1
            signals_per_day_std = len(events_std) / max(1, duration_days)
            
            # Adaptive generation
            if hasattr(gen, 'generate_adaptive'):
                if name in ['ATRShock', 'Microstructure', 'VWAPReversion', 'KalmanRegime']:
                    events_adj = gen.generate_adaptive(df, target_signals_per_day, **params)
                else:
                    events_adj = gen.generate_adaptive(df['close'], target_signals_per_day, **params)
                
                signals_per_day_adj = len(events_adj) / max(1, duration_days)
                
                print(f"{name:15s}: {signals_per_day_std:5.1f} -> {signals_per_day_adj:5.1f} signals/day "
                      f"({len(events_std):3d} -> {len(events_adj):3d} events)")
            else:
                print(f"{name:15s}: {signals_per_day_std:5.1f} signals/day ({len(events_std):3d} events) - No adaptive")
                
        except Exception as e:
            print(f"{name:15s}: Error - {e}")
    
    # Test full pipeline
    print(f"\n=== Full Pipeline Test ===")
    try:
        # Test without adaptive
        geoms_no_adaptive = orthogonal_label_generation(
            df['close'], 
            df['volume'], 
            df, 
            target_signals_per_day=target_signals_per_day,
            use_adaptive_thresholds=False
        )
        
        # Test with adaptive
        geoms_adaptive = orthogonal_label_generation(
            df['close'], 
            df['volume'], 
            df, 
            target_signals_per_day=target_signals_per_day,
            use_adaptive_thresholds=True
        )
        
        print(f"Non-adaptive: {len(geoms_no_adaptive)} geometries generated")
        print(f"Adaptive:     {len(geoms_adaptive)} geometries generated")
        
        # Show signal rates for adaptive results
        if geoms_adaptive:
            print("\nAdaptive generation signal rates:")
            for geom in geoms_adaptive[:5]:  # Show first 5
                duration_days = (geom.events[-1] - geom.events[0]).days if len(geom.events) > 1 else 1
                signals_per_day = len(geom.events) / max(1, duration_days)
                print(f"  {geom.family:20s}: {signals_per_day:5.1f} signals/day ({len(geom.events):3d} events)")
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_adaptive_thresholds()
