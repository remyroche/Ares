#!/usr/bin/env python3
"""
Test script for data quality decorators.
Demonstrates how the decorators automatically validate data quality for feature engineering.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

# Add the project root to the Python path
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.data_quality_decorators import (
    validate_data_quality,
    validate_wavelet_data_quality,
    validate_microstructure_data_quality,
    validate_klines_data_quality,
    ValidationLevel,
    clear_data_quality_cache,
    get_data_quality_cache_stats,
)


class TestFeatureEngineering:
    """Test class to demonstrate data quality decorators."""

    def __init__(self, symbol: str = "ETHUSDT", exchange: str = "BINANCE"):
        self.symbol = symbol
        self.exchange = exchange
        self.config = {"symbol": symbol, "exchange": exchange}

    @validate_data_quality(validation_level=ValidationLevel.WARNING)
    async def engineer_features(
        self, price_data: pd.DataFrame, volume_data: pd.DataFrame
    ) -> dict:
        """Main feature engineering method with automatic data quality validation."""
        print("✅ Feature engineering completed successfully!")
        return {"features": "engineered"}

    @validate_wavelet_data_quality
    async def analyze_wavelet_transforms(self, price_data: pd.DataFrame) -> dict:
        """Wavelet analysis with strict data quality validation."""
        print("✅ Wavelet analysis completed successfully!")
        return {"wavelet_features": "analyzed"}

    @validate_microstructure_data_quality
    async def analyze_microstructure_features(
        self, price_data: pd.DataFrame, volume_data: pd.DataFrame
    ) -> dict:
        """Microstructure analysis with volume data validation."""
        print("✅ Microstructure analysis completed successfully!")
        return {"microstructure_features": "analyzed"}

    @validate_klines_data_quality
    def calculate_price_impact(
        self, price_data: pd.DataFrame, volume_data: pd.DataFrame
    ) -> pd.Series:
        """Price impact calculation with OHLCV validation."""
        print("✅ Price impact calculation completed successfully!")
        return pd.Series([1.0, 2.0, 3.0], index=price_data.index[:3])


def create_test_data(
    data_type: str = "klines_ohlcv", quality: str = "good"
) -> pd.DataFrame:
    """Create test data of different types and quality levels."""

    # Create base timestamp index
    base_time = datetime(2024, 1, 1)
    timestamps = [base_time + timedelta(minutes=i) for i in range(1000)]

    if data_type == "klines_ohlcv":
        if quality == "good":
            # Good quality OHLCV data
            data = pd.DataFrame(
                {
                    "open": np.random.uniform(100, 200, 1000),
                    "high": np.random.uniform(100, 200, 1000),
                    "low": np.random.uniform(100, 200, 1000),
                    "close": np.random.uniform(100, 200, 1000),
                    "volume": np.random.uniform(1000, 10000, 1000),
                },
                index=timestamps,
            )
        elif quality == "bad":
            # Bad quality OHLCV data (negative prices, missing values)
            data = pd.DataFrame(
                {
                    "open": np.random.uniform(-50, 200, 1000),  # Negative values
                    "high": np.random.uniform(100, 200, 1000),
                    "low": np.random.uniform(100, 200, 1000),
                    "close": np.random.uniform(100, 200, 1000),
                    "volume": np.random.uniform(1000, 10000, 1000),
                },
                index=timestamps,
            )
            # Add some NaN values
            data.loc[data.index[100:150], "close"] = np.nan

    elif data_type == "aggtrades":
        # Aggregated trades data
        data = pd.DataFrame(
            {
                "price": np.random.uniform(100, 200, 1000),
                "quantity": np.random.uniform(1, 100, 1000),
                "trade_id": range(1000),
            },
            index=timestamps,
        )

    elif data_type == "futures_data":
        # Futures data with funding rate
        data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, 1000),
                "high": np.random.uniform(100, 200, 1000),
                "low": np.random.uniform(100, 200, 1000),
                "close": np.random.uniform(100, 200, 1000),
                "volume": np.random.uniform(1000, 10000, 1000),
                "funding_rate": np.random.uniform(-0.01, 0.01, 1000),
            },
            index=timestamps,
        )

    else:
        # Unknown format
        data = pd.DataFrame(
            {
                "unknown_col": np.random.randn(1000),
            },
            index=timestamps,
        )

    return data


async def test_data_quality_decorators():
    """Test the data quality decorators with different data types and quality levels."""

    print("🧪 Testing Data Quality Decorators")
    print("=" * 50)

    # Clear cache before testing
    clear_data_quality_cache()

    # Create test instance
    test_engineer = TestFeatureEngineering("ETHUSDT", "BINANCE")

    # Test 1: Good quality OHLCV data
    print("\n📊 Test 1: Good quality OHLCV data")
    print("-" * 30)
    good_data = create_test_data("klines_ohlcv", "good")
    volume_data = good_data[["volume"]].copy()
    price_data = good_data[["open", "high", "low", "close"]].copy()

    try:
        result = await test_engineer.engineer_features(price_data, volume_data)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Bad quality OHLCV data
    print("\n📊 Test 2: Bad quality OHLCV data")
    print("-" * 30)
    bad_data = create_test_data("klines_ohlcv", "bad")
    volume_data = bad_data[["volume"]].copy()
    price_data = bad_data[["open", "high", "low", "close"]].copy()

    try:
        result = await test_engineer.engineer_features(price_data, volume_data)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Wavelet analysis with good data
    print("\n📊 Test 3: Wavelet analysis with good data")
    print("-" * 30)
    try:
        result = await test_engineer.analyze_wavelet_transforms(price_data)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 4: Microstructure analysis
    print("\n📊 Test 4: Microstructure analysis")
    print("-" * 30)
    try:
        result = await test_engineer.analyze_microstructure_features(
            price_data, volume_data
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 5: Price impact calculation
    print("\n📊 Test 5: Price impact calculation")
    print("-" * 30)
    try:
        result = test_engineer.calculate_price_impact(price_data, volume_data)
        print(f"Result: {result.head()}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 6: Different data types
    print("\n📊 Test 6: Aggregated trades data")
    print("-" * 30)
    aggtrades_data = create_test_data("aggtrades")
    try:
        result = await test_engineer.engineer_features(aggtrades_data, aggtrades_data)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 7: Futures data
    print("\n📊 Test 7: Futures data")
    print("-" * 30)
    futures_data = create_test_data("futures_data")
    volume_data = futures_data[["volume"]].copy()
    price_data = futures_data[["open", "high", "low", "close"]].copy()

    try:
        result = await test_engineer.engineer_features(price_data, volume_data)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 8: Unknown data format
    print("\n📊 Test 8: Unknown data format")
    print("-" * 30)
    unknown_data = create_test_data("unknown")
    try:
        result = await test_engineer.engineer_features(unknown_data, unknown_data)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # Show cache statistics
    print("\n📊 Cache Statistics")
    print("-" * 30)
    cache_stats = get_data_quality_cache_stats()
    print(f"Cache size: {cache_stats['cache_size']}")
    print(f"Max size: {cache_stats['max_size']}")
    print(f"Cache keys: {len(cache_stats['cache_keys'])}")

    print("\n✅ Data quality decorator testing completed!")


if __name__ == "__main__":
    asyncio.run(test_data_quality_decorators())
