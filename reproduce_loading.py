
import os
import sys
import pandas as pd
from datetime import datetime
from src.utils.kline_parquet import KlinesParquetManager, StorageConfig
from src.launcher.ares_launcher import get_mode_lookback_days
from src.training.steps.market_analysis.shared_utils.execution_mode_lookback_config import get_execution_mode_config

def reproduce():
    print("--- REPRODUCING DATA LOADING ---")
    
    # Configuration mimicking BaseStep behavior for small_multi_asset
    symbol = "ETHUSDT"
    exchange = "binance"
    timeframe = "15m"
    execution_mode = "small_multi_asset"
    
    print(f"Target: {symbol} {exchange} {timeframe}")
    print(f"Mode: {execution_mode}")

    # Initialize Manager
    km = KlinesParquetManager(config=StorageConfig(base_dir="historical_data"))
    
    # 1. Check find_klines_files output
    print("\n[Step 1] Checking found files:")
    try:
        files = km._find_klines_files(symbol, exchange, timeframe, None, None, None)
        print(f"Found {len(files)} files.")
        if files:
            print(f"First: {files[0]}")
            print(f"Last: {files[-1]}")
    except Exception as e:
        print(f"Error finding files: {e}")

    # 2. Simulate default logic without start_date
    print("\n[Step 2] Resolving days_limit:")
    try:
        # Mimic BaseStep logic
        days_limit = None
        mode_days_defaults = {
            'light': 30, # simplified
            'blank': 180,
            'full': 1095
        }
        days_limit = mode_days_defaults.get(execution_mode)
        print(f"Initial days_limit (BaseStep map): {days_limit}")
        
        if days_limit is None:
            config = get_execution_mode_config()
            try:
                days_limit = config.get_data_loading_days(execution_mode)
                print(f"Config days_limit: {days_limit}")
            except Exception as e:
                print(f"Config lookup failed (expected): {e}")
                days_limit = 180 # Fallback 
    except Exception as e:
        print(f"Error resolving days: {e}")
    
    # 3. Load klines with fallback 180 
    print(f"\n[Step 3] Loading klines with last_n_days=180 (Fallback logic):")
    try:
        df = km.load_klines(
            symbol=symbol,
            exchange=exchange,
            interval=timeframe,
            last_n_days=180
        )
        if df is not None:
             print(f"Loaded DataFrame: {len(df)} rows")
             if not df.empty:
                 print(f"Time Range: {df.index.min()} to {df.index.max()}")
        else:
            print("Loaded None")
    except Exception as e:
        print(f"Load failed: {e}")

if __name__ == "__main__":
    reproduce()
