import pandas as pd
import sys

def inspect_parquet(path):
    print(f"--- Inspecting: {path} ---")
    try:
        df = pd.read_parquet(path)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Index: {df.index.dtype} ({df.index.min()} to {df.index.max()})")
        print("Head:")
        print(df.head(2))
        print("Tail:")
        print(df.tail(2))
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    except Exception as e:
        print(f"Error reading {path}: {e}")
    print("\n")

inspect_parquet("historical_data/binance/ethusdt/raw/ethusdt_1m_2021_12.parquet")
inspect_parquet("historical_data/binance/linkusdt/raw/linkusdt_1m_2023_03.parquet")
