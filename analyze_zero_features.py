#!/usr/bin/env python3
"""
Analyze zero values in HMM features to understand which specific features have the most zeros.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def analyze_zero_distribution():
    """Analyze the distribution of zero values in HMM features."""

    # Try to load the most recent data
    data_dir = Path("data_cache")
    possible_files = [
        data_dir / "klines_BINANCE_ETHUSDT_1m_2024-04.parquet",  # Most recent
        data_dir / "klines_BINANCE_ETHUSDT_1m_2023-06.parquet",
        data_dir / "binance_BTCUSDT_1m_aggtrades.parquet",
        data_dir / "binance_ETHUSDT_1m_aggtrades.parquet",
    ]

    df = None
    for file_path in possible_files:
        if file_path.exists():
            print(f"Loading data from {file_path}")
            df = pd.read_parquet(file_path)
            break

    if df is None or df.empty:
        print("No data files found!")
        return

    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check for timestamp column
    if 'timestamp' not in df.columns:
        print("No timestamp column found!")
        return

    # Convert timestamp if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Basic data validation
    print("\n=== BASIC DATA VALIDATION ===")
    print(f"Missing values per column:")
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"  {col}: {missing:,} missing values")

    # Check for zero values in raw data
    print("\n=== ZERO VALUES IN RAW DATA ===")
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            zeros = (df[col] == 0).sum()
            percentage = (zeros / len(df)) * 100
            print(f"  {col}: {zeros:,} zeros ({percentage:.3f}%)")

    # Simulate the feature engineering process
    print("\n=== SIMULATING FEATURE ENGINEERING ===")

    # Basic price features
    features = pd.DataFrame()
    features['timestamp'] = df['timestamp']

    # Momentum features
    print("Calculating momentum features...")
    for window in [5, 10, 20]:
        features[f'price_momentum_{window}'] = df['close'].pct_change(window)
        features[f'volume_momentum_{window}'] = df['volume'].pct_change(window)

    # RSI
    print("Calculating RSI...")
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    features['rsi'] = 100 - (100 / (1 + rs))

    # Volatility features
    print("Calculating volatility features...")
    price_returns = df['close'].pct_change()
    for window in [5, 10, 20]:
        features[f'volatility_{window}'] = price_returns.rolling(window=window).std()

    # Volume ratios
    print("Calculating volume features...")
    for window in [5, 10, 20]:
        volume_mean = df['volume'].rolling(window=window).mean()
        # Safe division
        features[f'volume_ratio_{window}'] = df['volume'] / volume_mean.replace(0, np.nan)

    # Technical indicators
    print("Calculating technical indicators...")
    features['sma_20'] = df['close'].rolling(window=20).mean()
    features['sma_50'] = df['close'].rolling(window=50).mean()

    # Distance calculations
    print("Calculating distance features...")
    features['price_vs_sma20'] = (df['close'] - features['sma_20']) / features['sma_20'].replace(0, np.nan)
    features['price_vs_sma50'] = (df['close'] - features['sma_50']) / features['sma_50'].replace(0, np.nan)

    # Drop timestamp for analysis
    feature_cols = [col for col in features.columns if col != 'timestamp']
    hmm_features = features[feature_cols]

    print(f"\nFeature matrix shape: {hmm_features.shape}")
    print(f"Number of features: {len(feature_cols)}")

    # Analyze zeros in features
    print("\n=== ZERO ANALYSIS IN FEATURES ===")

    total_zeros = (hmm_features == 0).sum().sum()
    print(f"Total zero values in all features: {total_zeros:,}")

    # Analyze each feature
    zero_counts = []
    for col in feature_cols:
        zeros = (hmm_features[col] == 0).sum()
        percentage = (zeros / len(hmm_features)) * 100
        zero_counts.append((col, zeros, percentage))

    # Sort by zero count
    zero_counts.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 20 features by zero count:")
    for i, (col, zeros, percentage) in enumerate(zero_counts[:20]):
        print(f"{i+1:2d}. {col}: {zeros:,} zeros ({percentage:.2f}%)")

    # Group by feature type
    print("\n=== ZERO ANALYSIS BY FEATURE TYPE ===")

    feature_types = {
        'momentum': [col for col in feature_cols if 'momentum' in col],
        'volatility': [col for col in feature_cols if 'volatility' in col],
        'volume': [col for col in feature_cols if 'volume' in col],
        'technical': [col for col in feature_cols if any(term in col for term in ['rsi', 'sma', 'price_vs'])],
    }

    for feature_type, cols in feature_types.items():
        if cols:
            type_zeros = sum((hmm_features[col] == 0).sum() for col in cols)
            type_percentage = (type_zeros / (len(hmm_features) * len(cols))) * 100
            print(f"{feature_type.capitalize():15}: {type_zeros:,} zeros ({type_percentage:.2f}%)")

    # Check for rows with many zeros
    print("\n=== ROW ANALYSIS ===")

    row_zero_counts = (hmm_features == 0).sum(axis=1)
    high_zero_rows = (row_zero_counts > 5).sum()
    print(f"Rows with more than 5 zeros: {high_zero_rows:,}")
    print(f"Max zeros in a single row: {row_zero_counts.max()}")

    # Check first vs rest
    first_100_zeros = (hmm_features.iloc[:100] == 0).sum().sum()
    rest_zeros = (hmm_features.iloc[100:] == 0).sum().sum()
    print(f"Zero values in first 100 rows: {first_100_zeros:,}")
    print(f"Zero values in remaining rows: {rest_zeros:,}")

if __name__ == "__main__":
    analyze_zero_distribution()
