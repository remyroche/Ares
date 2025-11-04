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


def generate_structural_market_state_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Generate structural market state features from OHLCV data.
    
    Categories:
    1. Liquidity/Microstructure: bid-ask spread proxy, volume imbalance, trade direction
    2. Trend-Convexity: slope-of-slope of MA, second derivative features
    3. Order flow persistence: buy volume bursts vs. sell volume bursts
    4. Regime flags: volatility percentile rank, session time blocks
    
    Klines provide: open, high, low, close, volume, quote_volume, trades, 
    taker_buy_base_volume, taker_buy_quote_volume
    """
    features = {}
    
    if len(df) < 5:
        return features
    
    try:
        # ===================================================================
        # CATEGORY 1: LIQUIDITY / MICROSTRUCTURE (OHLCV only)
        # ===================================================================
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
        
        # ===================================================================
        # CATEGORY 2: TREND-CONVEXITY (second derivatives)
        # ===================================================================
        
        # Slope-of-slope of MA (second derivative of moving average)
        if 'close' in df.columns:
            for window in [5, 10, 20]:
                if len(df) >= window + 2:
                    # Calculate MA
                    ma = df['close'].rolling(window=window).mean()
                    # Calculate first derivative (slope)
                    ma_slope = ma.diff()
                    # Calculate second derivative (slope-of-slope / acceleration)
                    ma_acceleration = ma_slope.diff()
                    if not ma_acceleration.empty and not pd.isna(ma_acceleration.iloc[-1]):
                        # Normalize by price to make scale-invariant
                        features[f'ma_acceleration_{window}'] = ma_acceleration.iloc[-1] / (df['close'].iloc[-1] + 1e-8)
        
        # Price second derivative (convexity measure)
        if 'close' in df.columns and len(df) >= 3:
            price_diff = df['close'].diff()
            price_second_diff = price_diff.diff()
            if not price_second_diff.empty and not pd.isna(price_second_diff.iloc[-1]):
                # Normalize by price
                features['price_convexity'] = price_second_diff.iloc[-1] / (df['close'].iloc[-1] + 1e-8)
        
        # Returns convexity (acceleration in returns)
        if 'close' in df.columns and len(df) >= 4:
            returns = df['close'].pct_change()
            returns_diff = returns.diff()  # First derivative of returns
            returns_accel = returns_diff.diff()  # Second derivative of returns
            if not returns_accel.empty and not pd.isna(returns_accel.iloc[-1]):
                features['returns_convexity'] = returns_accel.iloc[-1]
        
        # Volume convexity
        if 'volume' in df.columns and len(df) >= 3:
            vol_diff = df['volume'].diff()
            vol_second_diff = vol_diff.diff()
            if not vol_second_diff.empty and not pd.isna(vol_second_diff.iloc[-1]):
                avg_vol = df['volume'].mean()
                if avg_vol > 0:
                    features['volume_convexity'] = vol_second_diff.iloc[-1] / (avg_vol + 1e-8)
        
        # ===================================================================
        # CATEGORY 3: ORDER FLOW PERSISTENCE (OHLCV only - buy vs sell bursts)
        # ===================================================================
        
        # Buy volume burst detection (using taker_buy_base_volume)
        if 'taker_buy_base_volume' in df.columns and 'volume' in df.columns:
            for window in [3, 5, 10]:
                if len(df) >= window:
                    # Calculate buy/sell volume separation
                    taker_buy_vol = df['taker_buy_base_volume'].tail(window)
                    total_vol = df['volume'].tail(window)
                    taker_sell_vol = total_vol - taker_buy_vol
                    
                    # Buy volume burst: std dev of buy volume
                    buy_burst_std = taker_buy_vol.std()
                    if not pd.isna(buy_burst_std):
                        features[f'buy_volume_burst_{window}'] = buy_burst_std
                    
                    # Sell volume burst: std dev of sell volume
                    sell_burst_std = taker_sell_vol.std()
                    if not pd.isna(sell_burst_std):
                        features[f'sell_volume_burst_{window}'] = sell_burst_std
                    
                    # Buy/Sell burst ratio (which side is more volatile)
                    if sell_burst_std > 1e-8:
                        features[f'buy_sell_burst_ratio_{window}'] = buy_burst_std / sell_burst_std
                    
                    # Order flow persistence: autocorrelation of buy imbalance
                    buy_imbalance = (taker_buy_vol / (total_vol + 1e-8)) - 0.5  # Center at 0
                    if len(buy_imbalance) >= 2:
                        # Calculate autocorrelation
                        buy_imb_shifted = buy_imbalance.shift(1)
                        correlation = buy_imbalance.corr(buy_imb_shifted)
                        if not pd.isna(correlation):
                            features[f'order_flow_persistence_{window}'] = correlation
        
        # Price-volume burst coordination (do price and volume burst together?)
        if 'close' in df.columns and 'volume' in df.columns:
            for window in [5, 10]:
                if len(df) >= window:
                    price_changes = df['close'].pct_change().tail(window)
                    volume_changes = df['volume'].pct_change().tail(window)
                    
                    # Correlation between price moves and volume moves
                    correlation = price_changes.corr(volume_changes)
                    if not pd.isna(correlation):
                        features[f'price_volume_burst_sync_{window}'] = correlation
        
        # ===================================================================
        # CATEGORY 4: REGIME FLAGS
        # ===================================================================
        
        # Volatility percentile rank (where is current vol vs historical?)
        if 'close' in df.columns:
            for lookback in [20, 50, 100]:
                if len(df) >= lookback:
                    returns = df['close'].pct_change().dropna()
                    if len(returns) >= lookback:
                        # Current volatility (5-bar rolling std)
                        current_vol = returns.tail(5).std()
                        # Historical volatility distribution
                        historical_vols = returns.tail(lookback).rolling(5).std().dropna()
                        if len(historical_vols) > 0 and not pd.isna(current_vol):
                            # Percentile rank
                            percentile = (historical_vols < current_vol).sum() / len(historical_vols)
                            features[f'volatility_percentile_{lookback}'] = percentile
        
        # Session time blocks (intraday regime patterns)
        # For crypto: use UTC hour as proxy for session
        if hasattr(df.index, 'hour'):
            # Use last timestamp
            hour = df.index[-1].hour
            
            # Session blocks (4-hour blocks for crypto)
            features['session_block_0_4'] = 1 if 0 <= hour < 4 else 0  # Asia late/US late
            features['session_block_4_8'] = 1 if 4 <= hour < 8 else 0  # Asia morning
            features['session_block_8_12'] = 1 if 8 <= hour < 12 else 0  # Europe morning
            features['session_block_12_16'] = 1 if 12 <= hour < 16 else 0  # Europe afternoon/US morning
            features['session_block_16_20'] = 1 if 16 <= hour < 20 else 0  # US afternoon
            features['session_block_20_24'] = 1 if 20 <= hour < 24 else 0  # US evening/Asia early
            
            # Simplified: US vs non-US hours (higher liquidity proxy)
            features['is_us_session'] = 1 if 13 <= hour < 21 else 0  # US market hours in UTC
            features['is_asia_session'] = 1 if 0 <= hour < 9 else 0  # Asia market hours in UTC
            features['is_europe_session'] = 1 if 7 <= hour < 16 else 0  # Europe market hours in UTC
        
        # Rolling regime stability (how stable is volatility regime?)
        if 'close' in df.columns and len(df) >= 20:
            returns = df['close'].pct_change().dropna()
            if len(returns) >= 20:
                # Calculate rolling 5-bar volatility
                rolling_vol = returns.rolling(5).std()
                # Stability = inverse of volatility-of-volatility
                vol_of_vol = rolling_vol.tail(20).std()
                avg_vol = rolling_vol.tail(20).mean()
                if avg_vol > 1e-8 and not pd.isna(vol_of_vol):
                    # Low vol-of-vol = stable regime
                    features['regime_vol_stability'] = 1.0 - min(1.0, vol_of_vol / avg_vol)
        
        # Momentum regime flag (trending vs ranging)
        if 'close' in df.columns:
            for window in [10, 20]:
                if len(df) >= window:
                    returns = df['close'].pct_change().tail(window)
                    # Trending = consistent direction (high % positive or high % negative)
                    pct_positive = (returns > 0).sum() / len(returns)
                    # Ranging = balanced (close to 50/50)
                    trend_strength = abs(pct_positive - 0.5) * 2  # 0 = ranging, 1 = trending
                    features[f'momentum_regime_{window}'] = trend_strength
        
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
                
                # Add structural market state features (liquidity, convexity, order flow, regime flags)
                structural_features = generate_structural_market_state_features(chunk)
                regime_features.update(structural_features)
                
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
    
    # Quick dimensionality check before PCA
    print("\n   📈 Feature dimensionality check...")
    from sklearn.decomposition import PCA
    pca_check = PCA()
    pca_check.fit(feature_df_normalized.fillna(0))
    cumsum = np.cumsum(pca_check.explained_variance_ratio_)
    n_95 = np.argmax(cumsum >= 0.95) + 1
    print(f"      Components for 95% variance: {n_95}/{feature_df_normalized.shape[1]}")
    print(f"      Effective dimensionality: {100*n_95/feature_df_normalized.shape[1]:.1f}%")
    
    if n_95 < feature_df_normalized.shape[1] * 0.5:
        print(f"      ✅ Good: High intrinsic dimensionality")
    else:
        print(f"      ⚠️  Warning: Features may still have some redundancy")
    
    N_COMPONENTS = 15  # Match the component count from your HMM config
    
    # ENHANCEMENT: Categorize features before PCA for better regime detection
    print("\n   🔧 Categorizing features by type...")
    
    # Define feature categories based on economic significance
    feature_categories = {
        'structural': [],      # Liquidity, order flow, microstructure
        'volatility': [],      # Price variance, ranges
        'trend': [],           # Price direction, momentum  
        'volume': [],          # Volume patterns
        'momentum': [],        # Rate of change, acceleration
        'temporal': []         # Time-based patterns
    }
    
    # Categorize each feature by its name pattern
    for col in feature_df_normalized.columns:
        col_lower = col.lower()
        # Structural features (liquidity, order flow, microstructure) - MOST IMPORTANT
        if any(pattern in col_lower for pattern in [
            'order_flow', 'imbalance', 'buy_sell', 'price_impact',
            'trade_intensity', 'relative_spread', 'vw_price_range',
            'liquidity', 'microstructure', 'tick_imbalance'
        ]):
            feature_categories['structural'].append(col)
        # Volatility features
        elif any(pattern in col_lower for pattern in [
            'volatility', 'range', 'std', 'atr', 'variance'
        ]):
            feature_categories['volatility'].append(col)
        # Trend features
        elif any(pattern in col_lower for pattern in [
            'trend', 'ma', 'ema', 'price_to', 'temporal_price'
        ]):
            feature_categories['trend'].append(col)
        # Volume features
        elif any(pattern in col_lower for pattern in [
            'volume_ratio', 'volume_clustering', 'lagged_volume',
            'volume_momentum', 'volume_roc'
        ]):
            feature_categories['volume'].append(col)
        # Momentum features
        elif any(pattern in col_lower for pattern in [
            'momentum', 'roc', 'acceleration', 'velocity'
        ]):
            feature_categories['momentum'].append(col)
        # Temporal features
        elif any(pattern in col_lower for pattern in [
            'regime_duration', 'lagged_', 'temporal'
        ]):
            feature_categories['temporal'].append(col)
        else:
            # Default to structural if unknown
            feature_categories['structural'].append(col)
    
    print(f"      Structural: {len(feature_categories['structural'])} features")
    print(f"      Volatility: {len(feature_categories['volatility'])} features")
    print(f"      Trend: {len(feature_categories['trend'])} features")
    print(f"      Volume: {len(feature_categories['volume'])} features")
    print(f"      Momentum: {len(feature_categories['momentum'])} features")
    print(f"      Temporal: {len(feature_categories['temporal'])} features")
    
    # Apply PCA separately to structural features for HMM training
    structural_cols = feature_categories['structural']
    if len(structural_cols) == 0:
        print("      ⚠️ WARNING: No structural features found! Using all features as fallback.")
        structural_features = feature_df_normalized
    else:
        structural_features = feature_df_normalized[structural_cols]
    
    print(f"\n   🔧 Applying PCA to structural features only (for HMM training)...")
    print(f"      Input: {structural_features.shape[1]} structural features")
    
    # Fit PCA on structural features only
    pca_structural = PCA(n_components=N_COMPONENTS, random_state=42)
    structural_array_pca = pca_structural.fit_transform(structural_features.values)
    
    print(f"      ✅ Structural data reduced to {N_COMPONENTS} components")
    print(f"      Total variance explained: {np.sum(pca_structural.explained_variance_ratio_)*100:.2f}%")
    
    # Also apply PCA to all features for comparison/evaluation
    print(f"\n   🔧 Applying PCA to ALL features (for evaluation)...")
    pca_all = PCA(n_components=N_COMPONENTS, random_state=42)
    all_features_array_pca = pca_all.fit_transform(feature_df_normalized.values)
    print(f"      ✅ All features reduced to {N_COMPONENTS} components")
    print(f"      Total variance explained: {np.sum(pca_all.explained_variance_ratio_)*100:.2f}%")
    
    # Save features to cache
    cache_file = "hdp_hmm_features_cache.pkl"
    print(f"\n💾 Saving to cache: {cache_file}")
    pca_cols = [f'pca_{i}' for i in range(N_COMPONENTS)]
    
    # Save structural features (for HMM training)
    structural_df_f32 = pd.DataFrame(
        structural_array_pca.astype(np.float32), 
        columns=[f'structural_pca_{i}' for i in range(N_COMPONENTS)]
    )
    
    # Save all features (for evaluation)
    all_features_df_f32 = pd.DataFrame(
        all_features_array_pca.astype(np.float32),
        columns=[f'all_pca_{i}' for i in range(N_COMPONENTS)]
    )
    
    # Combine into one dataframe with clear naming
    cache_data = {
        'structural_features': structural_df_f32,  # For HMM training
        'all_features': all_features_df_f32,       # For evaluation
        'feature_categories': feature_categories,   # Category mapping
        'pca_structural': pca_structural,          # PCA transformer for structural
        'pca_all': pca_all                         # PCA transformer for all
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    cache_file_npy = "hdp_hmm_features_cache.npy"
    print(f"💾 Saving structural features to numpy cache: {cache_file_npy}")
    np.save(cache_file_npy, structural_array_pca.astype(np.float32))
    
    # ENHANCEMENT: Save price data for economic CV calculation
    print(f"\n💾 Saving price data for forward returns calculation...")
    price_cache_file = "hdp_hmm_price_cache.pkl"
    # Extract corresponding close prices and timestamps from df
    # We generated features using rolling windows, so align prices with features
    n_samples = len(structural_array_pca)
    price_data = {
        'close': df['close'].iloc[len(df) - n_samples:].values,
        'timestamp': df.index[len(df) - n_samples:].values if hasattr(df.index, 'values') else np.arange(n_samples)
    }
    with open(price_cache_file, 'wb') as f:
        pickle.dump(price_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"   ✅ Saved {len(price_data['close'])} price points")
    
    print("\n" + "=" * 80)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"   Structural features (HMM): {structural_array_pca.shape[1]} columns")
    print(f"   All features (eval): {all_features_array_pca.shape[1]} columns")
    print(f"   Samples: {structural_array_pca.shape[0]} rows")
    print(f"   Effective dims: {n_95} ({100*n_95/feature_df_normalized.shape[1]:.1f}%)")
    print(f"   Data type: float32")
    print(f"\n🚀 STRUCTURAL MARKET STATE FEATURES APPLIED:")
    print(f"   ✓ LIQUIDITY/MICROSTRUCTURE (OHLCV-based):")
    print(f"      - Bid-ask spread proxies (relative_spread)")
    print(f"      - Order flow imbalance (buy/sell pressure)")
    print(f"      - Trade direction & volume imbalance")
    print(f"   ✓ TREND-CONVEXITY (second derivatives):")
    print(f"      - MA acceleration (slope-of-slope)")
    print(f"      - Price/returns/volume convexity")
    print(f"   ✓ ORDER FLOW PERSISTENCE (OHLCV-based):")
    print(f"      - Buy vs sell volume burst measures")
    print(f"      - Order flow autocorrelation")
    print(f"      - Price-volume burst synchronization")
    print(f"   ✓ REGIME FLAGS:")
    print(f"      - Volatility percentile rank")
    print(f"      - Session time blocks (US/Asia/Europe)")
    print(f"      - Momentum regime indicators")
    print(f"   ✓ DATA QUALITY:")
    print(f"      - Dual-scale normalization (8h short + 48h long)")
    print(f"      - Correlation pruning (removed {len(useful_features) - feature_df_normalized.shape[1]} redundant features)")
    print(f"      - Feature categorization (structural vs volatility/trend/etc)")
    print(f"      - Separate PCA for structural features (prevents HMM 'cheating')")
    print(f"      - 50% memory reduction (float32)")
    print(f"      - Price data cached for economic CV calculation")
    print(f"\n📁 Cache files:")
    print(f"   - {cache_file}")
    print(f"   - {cache_file_npy}")
    print(f"   - {price_cache_file}")
    print(f"\n▶️  Now run the tuning script - it will load from cache!")
    print(f"\n🎯 KEY CHANGE: HMM will train on STRUCTURAL features only")
    print(f"   This prevents finding trivial regimes (e.g., 'high vol' vs 'low vol')")
    print(f"   All features still used for evaluation/analysis (CV ratios, etc.)")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

