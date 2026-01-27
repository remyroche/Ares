
import pandas as pd
import numpy as np
from src.utils.numba_funcs import _numba_generate_dollar_bars

def test_dollar_bars_timezone():
    # Create sample data with TZ-aware index
    dates = pd.date_range("2024-01-01", periods=1000, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open": np.random.randn(1000) + 100,
        "high": np.random.randn(1000) + 105,
        "low": np.random.randn(1000) + 95,
        "close": np.random.randn(1000) + 100,
        "volume": np.abs(np.random.randn(1000)) * 1000,
        "ca__feature": np.random.randn(1000)
    }, index=dates)

    # Simulate dollar bar generation
    threshold_vals = np.full(1000, 10000.0) # Threshold

    times_arr = df.index.values # int64 array of nanoseconds

    # Numba function
    db_times, db_opens, db_highs, db_lows, db_closes, db_vols = _numba_generate_dollar_bars(
        times_arr,
        df['open'].values, df['high'].values, df['low'].values, df['close'].values, df['volume'].values,
        threshold_vals
    )

    # Create dollar bar DF
    df_bars = pd.DataFrame({
        'close': db_closes
    }, index=pd.DatetimeIndex(db_times))

    print(f"Original Index TZ: {df.index.tz}")
    print(f"Dollar Bar Index TZ: {df_bars.index.tz}")

    # Try reindexing
    try:
        cross_asset = df[['ca__feature']].copy()
        aligned = cross_asset.reindex(df_bars.index, method='ffill')
        print(f"Aligned shape: {aligned.shape}")
        print(f"Aligned NaN count: {aligned.isna().sum().sum()}")
    except Exception as e:
        print(f"Reindex failed: {e}")

if __name__ == "__main__":
    test_dollar_bars_timezone()
