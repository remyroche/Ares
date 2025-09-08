#!/usr/bin/env python3
"""
Simple test script to verify the fixes for step06_advanced_features.py
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, '/Users/remyroche/Documents/Ares/src')

try:
    from training.steps.feature_engineering.step06_advanced_features import AdvancedFeatureEngineeringStep
    print("✅ Successfully imported AdvancedFeatureEngineeringStep")

    # Create mock config
    config = {
        "feature_engineering": {
            "enable_wavelets": False,
            "enable_multi_timeframe": True,
            "timeframes": ["5m", "15m", "1h"],
            "chunk_size": 10000,
            "max_features": 500,
            "feature_interaction_degree": 2,
            "regime_lookback_days": 30,
            "cross_timeframe_enabled": True,
            "regime_specific": True
        }
    }

    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)

    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    }, index=dates)

    print(f"📊 Created sample data with {len(data)} rows and columns: {list(data.columns)}")

    # Initialize the step
    step = AdvancedFeatureEngineeringStep(config)
    print("✅ Successfully initialized AdvancedFeatureEngineeringStep")

    # Test the problematic method
    try:
        result = step._cache_comprehensive_statistics(data)
        print("✅ Successfully ran _cache_comprehensive_statistics")
        print(f"📊 Cached {len(result)} technical indicators")
        print(f"🔍 Sample cached keys: {list(result.keys())[:10]}...")

        # Check if open_to_close is present
        if 'open_to_close' in result:
            print("✅ open_to_close feature is present in cached results")
        else:
            print("❌ open_to_close feature is missing from cached results")

    except Exception as e:
        print(f"❌ Error running _cache_comprehensive_statistics: {e}")
        import traceback
        traceback.print_exc()

    print("\n🎉 Test completed successfully!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
