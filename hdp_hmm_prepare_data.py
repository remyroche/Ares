#!/usr/bin/env python3
"""
Pre-compute and cache data for HDP-HMM tuning
This runs ONCE to avoid reloading data 810 times

IMPROVEMENTS:
1. Correlation pruner to remove redundant features
2. Keep both short (8h) and long (48h) variants
3. Increased long-term window from 32h to 48h
4. Order flow imbalance & microstructure features (klines-based)
"""

import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime, timedelta
import pickle
from typing import Dict, Tuple

print("=" * 80)
print("HDP-HMM Data Preparation - Enhanced Feature Engineering")
print("=" * 80)


def generate_microstructure_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Generate order flow imbalance and microstructure features from klines.
    
    Klines provide: open, high, low, close, volume, quote_volume, trades, 
    taker_buy_base_volume, taker_buy_quote_volume
    """
    features = {}
    
    if len(df) < 5:
        return features
    
    try:
        # Order Flow Imbalance (from taker buy/sell volumes)
        if 'taker_buy_base_volume' in df.columns and 'volume' in df.columns:
            taker_buy = df['taker_buy_base_volume'].iloc[-1]
            total_volume = df['volume'].iloc[-1]
            if total_volume > 0:
                # Positive = buy pressure, Negative = sell pressure
                features['order_flow_imbalance'] = (2 * taker_buy / total_volume) - 1
                
                # Rolling order flow imbalance (5, 10, 20 bars)
                for window in [5, 10, 20]:
                    if len(df) >= window:
                        taker_buy_sum = df['taker_buy_base_volume'].tail(window).sum()
                        volume_sum = df['volume'].tail(window).sum()
                        if volume_sum > 0:
                            features[f'order_flow_imbalance_{window}'] = (2 * taker_buy_sum / volume_sum) - 1
                
                # Order flow momentum (change in imbalance)
                if len(df) >= 10:
                    recent_imb = df['taker_buy_base_volume'].tail(5).sum() / df['volume'].tail(5).sum()
                    past_imb = df['taker_buy_base_volume'].iloc[-10:-5].sum() / df['volume'].iloc[-10:-5].sum()
                    features['order_flow_momentum'] = recent_imb - past_imb
        
        # Price Impact Proxy (price move per unit volume)
        if 'close' in df.columns and 'volume' in df.columns:
            price_change = df['close'].pct_change()
            volume = df['volume']
            
            for window in [5, 10]:
                if len(df) >= window:
                    price_move = abs(price_change.tail(window).sum())
                    avg_volume = volume.tail(window).mean()
                    if avg_volume > 0:
                        # Higher = more price impact per unit volume (less liquid)
                        features[f'price_impact_{window}'] = price_move / (avg_volume / volume.tail(window).std() if volume.tail(window).std() > 0 else 1)
        
        # Volume-Weighted Price Range (microstructure measure)
        if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            for window in [5, 10, 20]:
                if len(df) >= window:
                    price_range = (df['high'] - df['low']).tail(window)
                    volumes = df['volume'].tail(window)
                    if volumes.sum() > 0:
                        vw_range = (price_range * volumes).sum() / volumes.sum()
                        avg_close = df['close'].tail(window).mean()
                        if avg_close > 0:
                            features[f'vw_price_range_{window}'] = vw_range / avg_close
        
        # Trade Intensity (trades per unit volume)
        if 'trades' in df.columns and 'volume' in df.columns:
            for window in [5, 10]:
                if len(df) >= window:
                    trades = df['trades'].tail(window).sum()
                    volume = df['volume'].tail(window).sum()
                    if volume > 0:
                        features[f'trade_intensity_{window}'] = trades / volume
        
        # Relative Spread Proxy (high-low range as % of close)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            for window in [1, 5, 10]:
                if len(df) >= window:
                    avg_spread = ((df['high'] - df['low']) / df['close']).tail(window).mean()
                    features[f'relative_spread_{window}'] = avg_spread
        
        # Volume Clustering (autocorrelation of volume)
        if 'volume' in df.columns and len(df) >= 20:
            volume_series = df['volume'].tail(20)
            volume_changes = volume_series.pct_change().dropna()
            if len(volume_changes) >= 10:
                autocorr = volume_changes.autocorr(lag=1)
                if not pd.isna(autocorr):
                    features['volume_clustering'] = autocorr
        
        # Buy/Sell Pressure Asymmetry
        if 'taker_buy_quote_volume' in df.columns and 'quote_volume' in df.columns:
            for window in [5, 10]:
                if len(df) >= window:
                    buy_pressure = df['taker_buy_quote_volume'].tail(window).sum()
                    total_value = df['quote_volume'].tail(window).sum()
                    if total_value > 0:
                        # Ranges from -1 (all sells) to +1 (all buys)
                        features[f'buy_sell_asymmetry_{window}'] = (2 * buy_pressure / total_value) - 1
        
        # Tick Direction Imbalance (proportion of up vs down closes)
        if 'close' in df.columns:
            for window in [5, 10, 20]:
                if len(df) >= window:
                    closes = df['close'].tail(window)
                    up_ticks = (closes.diff() > 0).sum()
                    down_ticks = (closes.diff() < 0).sum()
                    total_ticks = up_ticks + down_ticks
                    if total_ticks > 0:
                        features[f'tick_imbalance_{window}'] = (up_ticks - down_ticks) / total_ticks
        
    except Exception as e:
        pass  # Silently skip on errors
    
    return features


@njit(fastmath=True, cache=True)
def _prune_correlated_numba(corr_matrix, variance, threshold):
    """Numba-accelerated correlation pruner."""
    n_features = corr_matrix.shape[0]
    # Numba doesn't like sets, so we use a boolean array
    features_to_remove = np.zeros(n_features, dtype=np.bool_)

    for i in range(n_features):
        if features_to_remove[i]:
            continue

        for j in range(i + 1, n_features):
            if features_to_remove[j]:
                continue

            if corr_matrix[i, j] > threshold:
                # Remove the feature with lower variance
                if variance[i] < variance[j]:
                    features_to_remove[i] = True
                    break  # Move to next i
                else:
                    features_to_remove[j] = True

    return features_to_remove

def prune_correlated_features(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove highly correlated features to reduce redundancy.
    (Wrapper for the Numba-accelerated function)
    """
    print(f"\n🔍 Pruning features with correlation > {threshold}...")

    corr_matrix = df.corr().abs()
    variance = df.var()

    # Call the fast Numba function
    features_to_remove_mask = _prune_correlated_numba(
        corr_matrix.values, 
        variance.values, 
        threshold
    )

    features_to_remove = df.columns[features_to_remove_mask]

    print(f"   ❌ Removing {len(features_to_remove)} redundant features")
    remaining_features = [col for col in df.columns if col not in features_to_remove]

    return df[remaining_features]


# Load data
print("\n📊 Loading 180 days of data...")
try:
    from src.utils.data.klines_parquet import KlinesParquetManager
    from src.feature_generation.categories.regime_feature_integration import RegimeFeatureIntegration
    
    klines_manager = KlinesParquetManager(data_dir="historical_data", exchange="binance")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    df = klines_manager.read_data(
        symbol="ETHUSDT", interval="1h",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    if df is None or df.empty:
        raise ValueError("No data loaded")
    
    print(f"   ✅ Loaded: {df.shape}")
    
    # Generate features
    print("   🔄 Generating regime features...")
    regime_integrator = RegimeFeatureIntegration()
    
    print("   🔧 Using rolling window (50-sample window, 5-sample step)...")
    feature_chunks = []
    total_chunks = (len(df) - 50 + 1) // 5
    
    for i in range(0, len(df) - 50 + 1, 5):
        chunk = df.iloc[i:i+50]
        if len(chunk) >= 48:
            try:
                # Generate regime features
                regime_features = regime_integrator._generate_regime_features(chunk)
                
                # Add microstructure features
                microstructure_features = generate_microstructure_features(chunk)
                regime_features.update(microstructure_features)
                
                chunk_df = pd.DataFrame([regime_features]) if isinstance(regime_features, dict) else regime_features
                feature_chunks.append(chunk_df)
            except Exception as e:
                continue
        
        if (len(feature_chunks) % 100) == 0:
            print(f"      Progress: {len(feature_chunks)}/{total_chunks} chunks...")
    
    if not feature_chunks:
        raise ValueError("No feature chunks were generated")

    print(f"   ✅ Generated {len(feature_chunks)} feature chunks")
    feature_df = pd.concat(feature_chunks, ignore_index=True).fillna(0)
    
    # Convert to numeric
    for col in feature_df.columns:
        if feature_df[col].dtype == 'object':
            try:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
            except:
                feature_df[col] = pd.Categorical(feature_df[col]).codes
    feature_df = feature_df.fillna(0)
    
    print(f"   📊 Raw features: {feature_df.shape[1]} columns")
    
    # NEW: Two-scale normalization with BOTH short and long windows
    print("   🔧 Applying two-scale normalization (8h and 48h) with Numpy vectorization...")
    
    # Convert to numpy for fast operations
    feature_values = feature_df.values
    n_samples, n_features = feature_values.shape
    
    # Create empty arrays for results
    z_short_values = np.zeros_like(feature_values)
    z_long_values = np.zeros_like(feature_values)
    
    # Define windows
    windows = [(8, 3), (48, 12)] # (window_size, min_periods)
    result_arrays = [z_short_values, z_long_values]
    prefixes = ['_short', '_long']
    
    with np.errstate(divide='ignore', invalid='ignore'):
        for i, (win, min_p) in enumerate(windows):
            # Create a 3D view of the data (n_samples, n_features, window_size)
            # This is an advanced, memory-efficient way to get rolling windows
            shape = (n_samples - win + 1, n_features, win)
            strides = (feature_values.strides[0], feature_values.strides[1], feature_values.strides[0])
            rolling_view = np.lib.stride_tricks.as_strided(feature_values, shape=shape, strides=strides)

            # Calculate rolling mean and std in a single vectorized operation
            rolling_mean = np.mean(rolling_view, axis=2)
            rolling_std = np.std(rolling_view, axis=2)

            # Apply z-score
            z_score_all = (feature_values[win-1:] - rolling_mean) / (rolling_std + 1e-8)
            
            # Pad the beginning (where window was too small)
            result_arrays[i][win-1:] = z_score_all
            
            # Handle min_periods (fill the start)
            if min_p < win:
                # Use a simpler expanding window for the start
                for r in range(min_p, win):
                    expanding_data = feature_values[:r]
                    mean = np.mean(expanding_data, axis=0)
                    std = np.std(expanding_data, axis=0)
                    z = (feature_values[r-1] - mean) / (std + 1e-8)
                    result_arrays[i][r-1] = z
    
    # Combine results back into a DataFrame
    feature_df_normalized = pd.DataFrame(
        np.concatenate([z_short_values, z_long_values], axis=1),
        columns=[f"{col}_short" for col in feature_df.columns] + [f"{col}_long" for col in feature_df.columns]
    )
    
    feature_df_normalized = feature_df_normalized.fillna(0).replace([np.inf, -np.inf], 0)
    
    print(f"   📊 After normalization: {feature_df_normalized.shape[1]} columns")
    
    # STEP 1: Remove zero-variance features
    print("\n   🔧 Step 1: Filtering zero-variance features...")
    feature_variances = feature_df_normalized.var()
    useful_features = feature_variances[feature_variances > 0.01].index
    removed_variance = len(feature_df_normalized.columns) - len(useful_features)
    print(f"      ❌ Removed {removed_variance} low-variance features")
    feature_df_normalized = feature_df_normalized[useful_features]
    print(f"      ✅ {len(useful_features)} features remain")
    
    # STEP 2: Prune highly correlated features
    print("\n   🔧 Step 2: Pruning correlated features (threshold=0.95)...")
    feature_df_normalized = prune_correlated_features(feature_df_normalized, threshold=0.95)
    print(f"      ✅ {feature_df_normalized.shape[1]} features remain after correlation pruning")
    
    # STEP 3: Drop rows with excessive zeros (warm-up artifacts)
    print("\n   🔧 Step 3: Removing zero-heavy rows (warm-up artifacts)...")
    zero_rate_per_row = (feature_df_normalized == 0).mean(axis=1)
    clean_rows = feature_df_normalized[zero_rate_per_row < 0.5]
    removed_rows = len(feature_df_normalized) - len(clean_rows)
    print(f"      ❌ Removed {removed_rows} zero-heavy rows")
    feature_df_normalized = clean_rows
    print(f"      ✅ {len(clean_rows)} samples remain")
    print("\n   🔧 Step 4: Applying PCA and caching reduced features...")
    from sklearn.decomposition import PCA
    
    N_COMPONENTS = 15 # Match the component count from your HMM config
    
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    
    # Fit PCA on the normalized data
    feature_array_pca = pca.fit_transform(feature_df_normalized.values)
    
    print(f"      ✅ Data reduced from {feature_df_normalized.shape[1]} to {N_COMPONENTS} components")
    print(f"      Total variance explained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

    # Quick PCA analysis for dimensionality check
    print("\n   📈 Feature dimensionality check...")
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(feature_df_normalized.fillna(0))
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.argmax(cumsum >= 0.95) + 1
    print(f"      Components for 95% variance: {n_95}/{feature_df_normalized.shape[1]}")
    print(f"      Effective dimensionality: {100*n_95/feature_df_normalized.shape[1]:.1f}%")
    
    if n_95 < feature_df_normalized.shape[1] * 0.5:
        print(f"      ✅ Good: High intrinsic dimensionality")
    else:
        print(f"      ⚠️  Warning: Features may still have some redundancy")
    
    print(f"\n   ✅ Final shape: {feature_df_normalized.shape}")
    
    # Convert to float32
    print("\n   🔧 Converting to float32 (50% memory savings)...")
    feature_array_f32 = feature_array_pca.astype(np.float32) # <-- NEW
    
    print(f"      Float64: {feature_df_normalized.values.nbytes / 1024:.2f} KB")
    print(f"      Float32: {feature_array_f32.nbytes / 1024:.2f} KB")
    print(f"      Savings: {(1 - feature_array_f32.nbytes/feature_df_normalized.values.nbytes)*100:.1f}%")
    
    # Save features to cache
    cache_file = "hdp_hmm_features_cache.pkl"
    print(f"\n💾 Saving to cache: {cache_file}")
    pca_cols = [f'pca_{i}' for i in range(N_COMPONENTS)]
    feature_df_f32 = pd.DataFrame(feature_array_f32, columns=pca_cols) # <-- NEW    with open(cache_file, 'wb') as f:
        pickle.dump(feature_df_f32, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    cache_file_npy = "hdp_hmm_features_cache.npy"
    print(f"💾 Saving to numpy cache: {cache_file_npy}")
    np.save(cache_file_npy, feature_array_f32)
    
    # ENHANCEMENT: Save price data for economic CV calculation
    print(f"\n💾 Saving price data for forward returns calculation...")
    price_cache_file = "hdp_hmm_price_cache.pkl"
    # Extract corresponding close prices and timestamps from df
    # We generated features using rolling windows, so align prices with features
    price_data = {
        'close': df['close'].iloc[len(df) - len(feature_array_f32):].values,
        'timestamp': df.index[len(df) - len(feature_array_f32):].values if hasattr(df.index, 'values') else np.arange(len(feature_array_f32))
    }
    with open(price_cache_file, 'wb') as f:
        pickle.dump(price_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   ✅ Saved {len(price_data['close'])} price points")
    
    print("\n" + "=" * 80)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"   Final features: {feature_array_f32.shape[1]} columns")
    print(f"   Samples: {feature_array_f32.shape[0]} rows")
    print(f"   Effective dims: {n_95} ({100*n_95/feature_array_f32.shape[1]:.1f}%)")
    print(f"   Data type: float32")
    print(f"\n🚀 IMPROVEMENTS APPLIED:")
    print(f"   ✓ Order flow imbalance features (klines-based)")
    print(f"   ✓ Microstructure features (volume, spread, price impact)")
    print(f"   ✓ Dual-scale normalization (8h short + 48h long)")
    print(f"   ✓ Correlation pruning (removed {len(useful_features) - feature_df_normalized.shape[1]} redundant features)")
    print(f"   ✓ 50% memory reduction (float32)")
    print(f"   ✓ Price data cached for economic CV calculation")
    print(f"\n📁 Cache files:")
    print(f"   - {cache_file}")
    print(f"   - {cache_file_npy}")
    print(f"   - {price_cache_file}")
    print(f"\n▶️  Now run the tuning script - it will load from cache!")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

