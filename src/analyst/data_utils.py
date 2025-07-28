# src/analyst/data_utils.py
import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks # For volume profile peaks

def load_klines_data(filename):
    """Loads k-line data from a CSV file."""
    if not os.path.exists(filename):
        print(f"Error: K-lines data file not found at {filename}")
        return pd.DataFrame()
    df = pd.read_csv(filename, index_col='open_time', parse_dates=True)
    df = df[~df.index.duplicated(keep='first')] # Remove duplicates
    # Ensure numeric columns are actually numeric
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_agg_trades_data(filename):
    """Loads aggregated trades data from a CSV file."""
    if not os.path.exists(filename):
        print(f"Error: Agg trades data file not found at {filename}")
        return pd.DataFrame()
    df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
    df = df[~df.index.duplicated(keep='first')] # Remove duplicates
    numeric_cols = ['price', 'quantity']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def load_futures_data(filename):
    """Loads futures data (funding rates, open interest) from a CSV file."""
    if not os.path.exists(filename):
        print(f"Error: Futures data file not found at {filename}")
        return pd.DataFrame()
    df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
    df = df[~df.index.duplicated(keep='first')] # Remove duplicates
    numeric_cols = ['fundingRate', 'openInterest']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def simulate_order_book_data(current_price):
    """Simulates real-time order book data for demonstration."""
    simulated_bids = [
        [current_price - 0.1, 5],
        [current_price - 0.2, 10],
        [current_price - 0.5, 20],
        [current_price - 1.0, 100],
        [current_price - 2.0, 50000 / current_price], # Large buy wall (approx $50k)
        [current_price - 2.5, 15],
        [current_price - 3.0, 120000 / current_price] # Even larger buy wall (approx $120k)
    ]
    simulated_asks = [
        [current_price + 0.1, 7],
        [current_price + 0.2, 12],
        [current_price + 0.5, 25],
        [current_price + 1.0, 80],
        [current_price + 2.0, 60000 / current_price], # Large sell wall (approx $60k)
        [current_price + 2.5, 18],
        [current_price + 3.0, 110000 / current_price] # Even larger sell wall (approx $110k)
    ]
    return {"bids": simulated_bids, "asks": simulated_asks}

def calculate_volume_profile(klines_df: pd.DataFrame, num_bins: int = 100):
    """
    Calculates Volume Profile (HVNs, LVNs, POC) for the given price range.
    Uses 'High', 'Low', 'Volume' from klines data.
    :param klines_df: DataFrame with 'High', 'Low', 'Volume' columns.
    :param num_bins: Number of price bins for the volume profile.
    :return: dict with 'poc', 'hvn_levels', 'lvn_levels', 'volume_in_bins' (Series with bin midpoints as index)
    """
    if klines_df.empty:
        return {"poc": np.nan, "hvn_levels": [], "lvn_levels": [], "volume_in_bins": pd.Series()}

    min_price = klines_df['Low'].min()
    max_price = klines_df['High'].max()
    
    if max_price == min_price: # Handle flat market
        return {"poc": min_price, "hvn_levels": [min_price], "lvn_levels": [], "volume_in_bins": pd.Series([klines_df['Volume'].sum()], index=[min_price])}

    # Create bins and sum volume within each bin
    # We'll create bins based on the overall price range
    bins = np.linspace(min_price, max_price, num_bins + 1)
    
    # To accurately assign volume to price bins, we can iterate through candles
    # and distribute their volume across the bins they span.
    # For simplicity and performance with OHLCV, we'll assign candle's volume to its midpoint bin.
    # A more precise method would involve distributing volume proportionally across price ranges.
    
    # Assign each candle's midpoint to a bin and sum its volume
    mid_prices = (klines_df['High'] + klines_df['Low']) / 2
    
    # Use pd.cut to categorize each midpoint into a bin interval
    price_bins_categorized = pd.cut(mid_prices, bins, include_lowest=True)
    
    # Group by these categories and sum volume
    volume_profile_series = klines_df.groupby(price_bins_categorized)['Volume'].sum()
    
    # Map bin intervals to their midpoints for a more usable index
    bin_midpoints_map = {interval: (interval.left + interval.right) / 2 for interval in volume_profile_series.index}
    volume_profile = volume_profile_series.rename(index=bin_midpoints_map)
    volume_profile = volume_profile.fillna(0) # Fill bins with no volume as 0

    # Point of Control (POC): Price level (midpoint of bin) with highest volume
    poc_price = volume_profile.idxmax() if not volume_profile.empty else np.nan

    # High-Volume Nodes (HVNs): Peaks in the volume profile
    # Find peaks in the volume_profile series values
    # prominence: minimum height of a peak relative to its neighbors
    # width: minimum number of data points in a peak
    hvn_indices, _ = find_peaks(volume_profile.values, prominence=volume_profile.max() * 0.1, width=3) # 10% prominence, min 3 bins wide
    hvn_levels = [volume_profile.index[i] for i in hvn_indices]

    # Low-Volume Nodes (LVNs): Troughs in the volume profile
    # Find peaks in the *negative* volume_profile values
    lvn_indices, _ = find_peaks(-volume_profile.values, prominence=volume_profile.max() * 0.05, width=3) # 5% prominence for troughs
    lvn_levels = [volume_profile.index[i] for i in lvn_indices]
    
    # Sort levels for consistency
    hvn_levels.sort()
    lvn_levels.sort()

    # print(f"Volume Profile: POC={poc_price:.2f}, HVNs={hvn_levels}, LVNs={lvn_levels}")
    return {"poc": poc_price, "hvn_levels": hvn_levels, "lvn_levels": lvn_levels, "volume_in_bins": volume_profile}

def create_dummy_data(filename, data_type, num_records=1000, start_date='2023-01-01'):
    """
    Creates dummy CSV data for klines, aggregated trades, or futures.
    This function is now centralized in data_utils.
    """
    if os.path.exists(filename):
        print(f"Dummy data file '{filename}' already exists. Skipping creation.")
        return

    print(f"Creating dummy {data_type} data at {filename}...")
    dates = pd.date_range(start=start_date, periods=num_records, freq='1min')

    if data_type == 'klines':
        # Simulate price movement
        price = 1000 + np.cumsum(np.random.randn(num_records))
        df = pd.DataFrame({
            'open_time': dates,
            'open': price,
            'high': price + np.random.rand(num_records) * 5,
            'low': price - np.random.rand(num_records) * 5,
            'close': price + np.random.randn(num_records),
            'volume': np.random.randint(100, 10000, num_records),
            'close_time': dates + pd.Timedelta(minutes=1),
            'quote_asset_volume': np.random.rand(num_records) * 100000,
            'number_of_trades': np.random.randint(50, 500, num_records),
            'taker_buy_base_asset_volume': np.random.rand(num_records) * 5000,
            'taker_buy_quote_asset_volume': np.random.rand(num_records) * 50000,
            'ignore': 0
        })
        df.set_index('open_time', inplace=True)
    elif data_type == 'agg_trades':
        df = pd.DataFrame({
            'timestamp': dates,
            'a': np.arange(num_records), # Aggregate tradeId
            'p': 1000 + np.cumsum(np.random.randn(num_records) * 0.1), # Price
            'q': np.random.rand(num_records) * 10, # Quantity
            'f': np.arange(num_records), # First tradeId
            'l': np.arange(num_records), # Last tradeId
            'T': (dates.astype(np.int64) // 10**6), # Timestamp in ms
            'm': np.random.choice([True, False], num_records), # Was the buyer the maker?
            'M': np.random.choice([True, False], num_records)  # Was the trade the best price match?
        })
        df.set_index('timestamp', inplace=True)
        df.rename(columns={'p': 'price', 'q': 'quantity', 'm': 'is_buyer_maker'}, inplace=True)
    elif data_type == 'futures':
        df = pd.DataFrame({
            'timestamp': dates,
            'fundingRate': np.random.rand(num_records) * 0.0001 - 0.00005, # Small positive/negative
            'openInterest': np.random.rand(num_records) * 1000000 + 500000 # Large numbers
        })
        df.set_index('timestamp', inplace=True)
    else:
        print(f"Unknown data type: {data_type}. Skipping dummy data creation.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True) # Ensure directory exists
    df.to_csv(filename)
    print(f"Dummy {data_type} data saved to '{filename}'.")

