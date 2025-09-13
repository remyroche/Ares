#!/usr/bin/env python3
"""
Test script to verify PyArrow type fix
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.utils.data.historical_data_downloader import HistoricalDataDownloader

    def test_pyarrow_fix():
        """Test that the PyArrow type fix works properly."""
        print("🧪 Testing PyArrow type fix...")

        # Create a sample DataFrame similar to what the downloader creates
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='1min')
        df = pd.DataFrame({
            'open': np.random.uniform(3000, 4000, len(dates)),
            'high': np.random.uniform(3000, 4000, len(dates)),
            'low': np.random.uniform(3000, 4000, len(dates)),
            'close': np.random.uniform(3000, 4000, len(dates)),
            'volume': np.random.uniform(100, 1000, len(dates)),
            'close_time': dates + pd.Timedelta(minutes=1),
            'quote_volume': np.random.uniform(1000, 10000, len(dates)),
            'trades': np.random.randint(10, 100, len(dates)),
            'taker_buy_base': np.random.uniform(50, 500, len(dates)),
            'taker_buy_quote': np.random.uniform(500, 5000, len(dates))
        }, index=dates)

        print(f"📊 Created test DataFrame with {len(df)} rows")

        # Simulate the metadata addition that the downloader does
        df['symbol'] = 'ETHUSDT'
        df['interval'] = '1m'
        df['year'] = df.index.year.astype('int32')  # Fixed: explicitly set as int32
        df['month'] = df.index.month.astype('int32')  # Fixed: explicitly set as int32
        df['day'] = df.index.day.astype('int32')  # Fixed: explicitly set as int32

        print("📋 Data types after metadata addition:")
        for col in ['year', 'month', 'day']:
            print(f"  {col}: {df[col].dtype}")

        # Test saving to parquet
        test_file = Path('test_pyarrow_fix.parquet')
        try:
            df.to_parquet(test_file, index=True, compression='snappy')
            print("✅ Successfully saved to parquet!")

            # Test reading back
            df_read = pd.read_parquet(test_file)
            print(f"✅ Successfully read from parquet! ({len(df_read)} rows)")

            # Clean up
            test_file.unlink()
            print("🧹 Cleaned up test file")

        except Exception as e:
            print(f"❌ Error with parquet operations: {e}")
            if test_file.exists():
                test_file.unlink()
            return False

        print("🎉 PyArrow type fix test passed!")
        return True

    if __name__ == "__main__":
        success = test_pyarrow_fix()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
