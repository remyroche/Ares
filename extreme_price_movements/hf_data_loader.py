"""
High-frequency (15m) OHLCV data loader for precise trailing profit simulation.

Downloads 15m data via CCXT and stores in parquet format.
Checks local storage before downloading to avoid redundant API calls.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import ccxt
from extreme_price_movements.utils import tprint


# Storage directory for 15m data
HF_DATA_DIR = Path(os.environ.get("EPM_HF_DATA_DIR", str(Path(__file__).parent / "15m_ohlcv")))
HF_DATA_DIR.mkdir(exist_ok=True)


def _get_parquet_path(symbol: str) -> Path:
    """Get parquet file path for a symbol."""
    # Normalize symbol: BTC/USDT -> btcusdt
    clean_symbol = symbol.replace("/", "").lower()
    return HF_DATA_DIR / f"{clean_symbol}_15m.parquet"


def _load_existing_data(symbol: str) -> pd.DataFrame:
    """Load existing 15m data from parquet if available."""
    path = _get_parquet_path(symbol)
    if path.exists():
        try:
            df = pd.read_parquet(path)
            idx = pd.to_datetime(df.index)
            if idx.tz is None:
                df.index = idx.tz_localize("UTC")
            else:
                df.index = idx.tz_convert("UTC")
            return df
        except Exception as e:
            tprint(f"WARNING: Failed to load {path}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def _save_data(symbol: str, df: pd.DataFrame):
    """Save 15m data to parquet with float32 downcasting."""
    if df.empty:
        return
    
    # Downcast to float32
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    
    path = _get_parquet_path(symbol)
    df.to_parquet(path, compression='snappy')


def _download_from_exchange(exchange: ccxt.Exchange, symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Download 15m OHLCV via CCXT."""
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    
    tprint(f"Downloading 15m data for {symbol}: {start_ts} to {end_ts}")
    
    try:
        # CCXT uses '15m' timeframe
        all_ohlcv = []
        current_ms = start_ms
        
        # Fetch in chunks (CCXT has limits per request)
        while current_ms < end_ms:
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe='15m',
                since=current_ms,
                limit=1000  # Max per request
            )
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Move to next chunk
            current_ms = ohlcv[-1][0] + 1
            
            # Stop if we've reached the end
            if ohlcv[-1][0] >= end_ms:
                break
        
        if not all_ohlcv:
            tprint(f"WARNING: No data returned for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')
        
        # Filter to requested range
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df
    
    except Exception as e:
        tprint(f"ERROR: Failed to download {symbol} 15m data: {e}")
        return pd.DataFrame()


def get_15m_ohlcv(exchange: ccxt.Exchange, symbol: str, entry_ts: pd.Timestamp, max_hold_hours: int = 12) -> pd.DataFrame:
    """
    Get 15m OHLCV data for a symbol, downloading if necessary.
    
    Downloads 12 hours of data at once to cover typical holding periods.
    Checks local parquet storage before downloading.
    Overwrites existing data if there's overlap.
    
    Args:
        exchange: CCXT exchange instance (e.g., ccxt.binance())
        symbol: Trading pair in CCXT format (e.g., 'BTC/USDT')
        entry_ts: Entry timestamp
        max_hold_hours: Hours of data to ensure available (default 12)
    
    Returns:
        DataFrame with 15m OHLCV data
    """
    # Ensure UTC
    if entry_ts.tz is None:
        entry_ts = entry_ts.tz_localize('UTC')
    else:
        entry_ts = entry_ts.tz_convert('UTC')
    
    # Download window: 12 hours from entry
    download_start = entry_ts
    download_end = entry_ts + pd.Timedelta(hours=max_hold_hours)
    
    # Load existing data
    existing_df = _load_existing_data(symbol)
    
    # Check if we need to download
    need_download = False
    
    if existing_df.empty:
        need_download = True
    else:
        # Check coverage
        existing_start = existing_df.index.min()
        existing_end = existing_df.index.max()
        
        # Need download if requested range is not fully covered
        if download_start < existing_start or download_end > existing_end:
            need_download = True
    
    if need_download:
        # Download new data
        new_df = _download_from_exchange(exchange, symbol, download_start, download_end)
        
        if not new_df.empty:
            if existing_df.empty:
                # No existing data, just save new
                combined_df = new_df
            else:
                # Merge with existing, overwriting overlaps
                combined_df = pd.concat([existing_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.sort_index()
            
            # Save updated data
            _save_data(symbol, combined_df)
            
            # Return requested range
            return combined_df.loc[entry_ts:download_end]
        else:
            # Download failed, return existing data if available
            if not existing_df.empty:
                return existing_df.loc[entry_ts:download_end]
            return pd.DataFrame()
    else:
        # Use existing data
        return existing_df.loc[entry_ts:download_end]


def clear_cache(symbol: str = None):
    """
    Clear cached 15m data.
    
    Args:
        symbol: If provided, clear only this symbol. Otherwise clear all.
    """
    if symbol:
        path = _get_parquet_path(symbol)
        if path.exists():
            path.unlink()
            tprint(f"Cleared 15m cache for {symbol}")
    else:
        for path in HF_DATA_DIR.glob("*_15m.parquet"):
            path.unlink()
        tprint("Cleared all 15m cache")
