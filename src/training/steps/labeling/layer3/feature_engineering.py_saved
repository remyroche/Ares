"""
Layer 3 Feature Engineering

Handles feature generation, optimization, and Layer 0 integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange
import numba as nb
import hashlib
from pathlib import Path
import pickle

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

logger = logging.getLogger(__name__)

EPS = 1e-12

@njit
def select_features_by_correlation_jit(corr_matrix: np.ndarray, target_corr: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """
    JIT-compiled feature selection based on correlation filtering.

    Args:
        corr_matrix: Correlation matrix between features
        target_corr: Correlation of each feature with target
        threshold: Correlation threshold for filtering

    Returns:
        Boolean array indicating which features to keep
    """
    n_features = corr_matrix.shape[0]
    features_to_keep = np.ones(n_features, dtype=np.bool_)

    # Find highly correlated pairs
    high_corr_mask = corr_matrix > threshold

    for i in range(n_features):
        for j in range(i + 1, n_features):  # Only check upper triangle
            if high_corr_mask[i, j] and features_to_keep[i] and features_to_keep[j]:
                # Keep the one with higher correlation to target
                if target_corr[i] < target_corr[j]:
                    features_to_keep[i] = False
                else:
                    features_to_keep[j] = False

    return features_to_keep

def _create_cache_key(df, market_data, operation_name):
    """Create a cache key based on data characteristics"""
    # Create a hash based on shape, column names, and some statistical properties
    key_components = [
        str(df.shape),
        str(sorted(df.columns.tolist())),
        str(df.shape[0]),  # number of rows
        f"{df.values.mean():.6f}",  # mean of all values
        f"{df.values.std():.6f}",   # std of all values
        operation_name
    ]

    if market_data is not None:
        key_components.extend([
            str(market_data.shape),
            str(sorted(market_data.columns.tolist())),
            f"{market_data.values.mean():.6f}",
            f"{market_data.values.std():.6f}"
        ])

    key_string = "|".join(key_components)
    return hashlib.md5(key_string.encode()).hexdigest()

def _get_cache_path(cache_key):
    """Get cache file path"""
    cache_dir = Path("cache/layer3_features")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.pkl"

def _load_from_cache(cache_key):
    """Load cached result if available"""
    cache_path = _get_cache_path(cache_key)
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            # Cache corrupted, remove it
            cache_path.unlink(missing_ok=True)
    return None

def _save_to_cache(cache_key, result):
    """Save result to cache"""
    try:
        cache_path = _get_cache_path(cache_key)
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
    except Exception:
        # Cache write failed, continue without caching
        pass

def enhance_layer3_features_optimized(df, market_data, layer1_weight, layer0_params=None, fast_mode=False):
    """
    Add optimized Layer 0 feature combinations, noise features, and Layer 1 weight integration.
    
    Uses intelligent combinations of Layer 0 filtering methods rather than individual features.

    Args:
        fast_mode: Skip expensive computations for faster execution
    """
    if fast_mode:
        tprint_info("⚡ Fast mode: minimal Layer 0 feature integration")
        # Add just basic momentum features
        if 'close' in market_data.columns:
            df['fast_momentum'] = market_data['close'].pct_change()
            df['fast_volatility'] = market_data['close'].rolling(20).std()
        return df

    tprint_info("🔧 Enhancing Layer 3 with Optimized Layer 0 Features...")
    tprint_info(f"📊 Input dataframe: {len(df)} rows, {len(df.columns)} columns")
    
    # Load Layer 0 parameters if not provided
    if layer0_params is None:
        try:
            from src.training.steps.labeling.unified_price_layer2 import load_layer0_params
            layer0_params = load_layer0_params()
            tprint_success("✅ Loaded Layer 0 parameters")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load Layer 0 params: {e}")
            layer0_params = {}
    
    # Show Layer 0 configuration
    enabled_features = []
    if layer0_params.get('adaptive_kalman_enabled', False):
        # Always enable adaptive Kalman for Layer 3
        enabled_features.append('Adaptive Kalman')
    if layer0_params.get('robust_vwap_enabled', False):
        enabled_features.append('Robust VWAP')
    if layer0_params.get('hampel_filter_enabled', False):
        # Always enable Hampel filter for Layer 3
        enabled_features.append('Hampel Filter')
    if layer0_params.get('savgol_filter_enabled', False):
        # Always enable Savitzky-Golay filter for Layer 3
        enabled_features.append('Savitzky-Golay')
    
    tprint_info(f"🔧 Layer 0 enabled features: {enabled_features}")
    
    initial_feature_count = len(df.columns)
    
    # === Optimized Layer 0 Feature Combinations ===
    
    # 1. Unified Filtered Price (combines all enabled methods)
    tprint_info("📈 Generating Unified Filtered Price...")
    try:
        # Check cache for unified price computation
        cache_key = _create_cache_key(df, market_data, "unified_price")
        cached_result = _load_from_cache(cache_key)

        if cached_result is not None:
            tprint_success("✅ Loaded unified price from cache")
            unified_price = cached_result
        else:
            from src.training.steps.labeling.unified_price_layer2 import generate_unified_layer2_price
            unified_price = generate_unified_layer2_price(market_data, layer0_params)
            unified_price = unified_price.reindex(df.index).fillna(method='ffill')
            # Cache the result
            _save_to_cache(cache_key, unified_price)
            tprint_success("✅ Computed and cached unified price")
        
        # Core unified price features (4 features)
        df['unified_price_momentum'] = unified_price.pct_change()
        df['unified_price_strength'] = (unified_price > unified_price.rolling(20).mean()).astype(int)
        df['unified_volatility_adj'] = unified_price.rolling(20).std() / market_data['volatility_1d']
        df['unified_regime_confidence'] = 1 - np.abs(unified_price - market_data['close']) / market_data['close']
        
        unified_stats = f"mean={unified_price.mean():.4f}, std={unified_price.std():.4f}"
        tprint_success(f"✅ Unified price: {unified_stats}")
        tprint_success(f"✅ Added 4 unified price features")
        
    except Exception as e:
        tprint_warning(f"⚠️ Unified price features failed: {e}")
    
    # 2. Adaptive Filtering Score (combines adaptive kalman + robust vwap)
    tprint_info("🔄 Generating Adaptive Filtering Features...")
    adaptive_features = []
    
    if layer0_params.get('adaptive_kalman_enabled', False):
        # Always enable adaptive Kalman for Layer 3
        try:
            from src.training.steps.labeling.unified_price_layer2 import generate_adaptive_kalman_price
            adaptive_price = generate_adaptive_kalman_price(
                market_data,
                base_Q=layer0_params.get('kalman_Q', 1e-4),
                base_R=layer0_params.get('kalman_R', 0.01),
                noise_window=layer0_params.get('adaptive_noise_window', 50),
                adaptation_rate=layer0_params.get('adaptive_adaptation_rate', 0.1)
            )
            adaptive_features.append(adaptive_price.reindex(df.index).fillna(method='ffill'))
            tprint_success("✅ Generated adaptive Kalman price")
        except Exception as e:
            tprint_warning(f"⚠️ Adaptive Kalman failed: {e}")
    
    if layer0_params.get('robust_vwap_enabled', False) and 'volume' in market_data.columns:
        # Always enable robust VWAP when volume available
        try:
            from src.training.steps.labeling.unified_price_layer2 import generate_robust_vwap_price
            robust_vwap = generate_robust_vwap_price(
                market_data,
                base_lookback=layer0_params.get('vwap_lookback', 50),
                min_lookback=layer0_params.get('robust_min_lookback', 20),
                max_lookback=layer0_params.get('robust_max_lookback', 200),
                volatility_window=layer0_params.get('robust_volatility_window', 20)
            )
            adaptive_features.append(robust_vwap.reindex(df.index).fillna(method='ffill'))
            tprint_success("✅ Generated robust VWAP price")
        except Exception as e:
            tprint_warning(f"⚠️ Robust VWAP failed: {e}")
    
    # Combine adaptive features (if multiple available)
    if len(adaptive_features) > 1:
        # Average of adaptive methods
        adaptive_combined = pd.concat(adaptive_features, axis=1).mean(axis=1)
        tprint_info(f"🔄 Combined {len(adaptive_features)} adaptive methods")
    elif len(adaptive_features) == 1:
        adaptive_combined = adaptive_features[0]
        tprint_info("🔄 Using single adaptive method")
    else:
        adaptive_combined = market_data['close'].reindex(df.index)  # Fallback
    tprint_warning("⚠️ No adaptive features available, using close price")
    
    # Adaptive filtering features (3 features)
    df['adaptive_filter_momentum'] = adaptive_combined.pct_change()
    df['adaptive_filter_distance'] = (market_data['close'] - adaptive_combined) / market_data['close']
    df['adaptive_filter_regime'] = (adaptive_combined > adaptive_combined.rolling(50).mean()).astype(int)
    
    tprint_success(f"✅ Added 3 adaptive filtering features")
    
    # 3. Noise Reduction Score (combines hampel + savgol)
    tprint_info("🔇 Generating Noise Reduction Features...")
    noise_reduction_features = []
    
    if layer0_params.get('hampel_filter_enabled', False):
        # Always enable Hampel filter for Layer 3
        try:
            from src.training.steps.labeling.unified_price_layer2 import apply_hampel_filter
            hampel_price = apply_hampel_filter(
                market_data['close'],
                window=layer0_params.get('hampel_window', 5),
                threshold=layer0_params.get('hampel_threshold', 3.0)
            )
            noise_reduction_features.append(hampel_price.reindex(df.index))
            tprint_success("✅ Generated Hampel filtered price")
        except Exception as e:
            tprint_warning(f"⚠️ Hampel filter failed: {e}")
    
    if layer0_params.get('savgol_filter_enabled', False):
        # Always enable Savitzky-Golay filter for Layer 3
        try:
            from src.training.steps.labeling.unified_price_layer2 import apply_savgol_filter
            savgol_price = apply_savgol_filter(
                market_data['close'],
                window_length=layer0_params.get('savgol_window', 21),
                poly_order=layer0_params.get('savgol_order', 3)
            )
            noise_reduction_features.append(savgol_price.reindex(df.index))
            tprint_success("✅ Generated Savitzky-Golay filtered price")
        except Exception as e:
            tprint_warning(f"⚠️ Savitzky-Golay filter failed: {e}")
    
    # Combine noise reduction features
    if len(noise_reduction_features) > 1:
        # Average of noise reduction methods
        noise_combined = pd.concat(noise_reduction_features, axis=1).mean(axis=1)
        tprint_info(f"🔇 Combined {len(noise_reduction_features)} noise reduction methods")
    elif len(noise_reduction_features) == 1:
        noise_combined = noise_reduction_features[0]
        tprint_info("🔇 Using single noise reduction method")
    else:
        noise_combined = market_data['close'].reindex(df.index)  # Fallback
    tprint_warning("⚠️ No noise reduction features available, using close price")
    
    # Noise reduction features (3 features)
    df['noise_reduction_momentum'] = noise_combined.pct_change()
    df['noise_reduction_smoothness'] = 1 - np.abs(noise_combined - market_data['close']) / market_data['close']
    df['noise_reduction_stability'] = (np.abs(noise_combined - market_data['close']) < 0.01).astype(int)
    
    tprint_success(f"✅ Added 3 noise reduction features")
    
    # 4. Filter Consensus Score (agreement across all methods)
    tprint_info("🤝 Calculating Filter Consensus...")
    all_filtered_prices = []
    # Ensure we have at least unified_price for consensus
    if 'unified_price' not in locals() or unified_price is None:
        unified_price = market_data['close'].reindex(df.index)
    if 'unified_price' in locals():
        all_filtered_prices.append(unified_price)
    all_filtered_prices.extend(adaptive_features)
    all_filtered_prices.extend(noise_reduction_features)
    
    if len(all_filtered_prices) >= 2:
        # Calculate consensus (agreement) across filtering methods
        filter_matrix = pd.concat(all_filtered_prices, axis=1)
        filter_consensus = filter_matrix.std(axis=1)  # Low std = high consensus
        df['filter_consensus_score'] = 1 - (filter_consensus / market_data['close'])
        # df['filter_disagreement_volatility'] = filter_consensus.rolling(20).std()
        
        consensus_stats = f"mean={df['filter_consensus_score'].mean():.3f}"
        tprint_success(f"✅ Filter consensus: {consensus_stats}")
        tprint_success(f"✅ Added 1 consensus feature")
    else:
        df['filter_consensus_score'] = 1.0  # Perfect consensus fallback
    # df['filter_disagreement_volatility'] = 0.0
    tprint_warning("⚠️ Insufficient filters for consensus, using fallback values")
    
    # === Advanced Noise Features (4 features) ===
    tprint_info("📊 Generating Advanced Noise Features...")
    
    # Price disorder score (market microstructure noise indicator)
    df['price_disorder_score'] = market_data['close'].rolling(20).std() / market_data['close'].rolling(100).std()
    
    # Note: Removed ensemble-derived features (signal_noise_ratio, volatility_normalized_noise, entropy_based_noise)
    # These features were using ensemble_prob, ens_uncertainty, and ens_prediction_dispersion
    # which are aggregations of base model predictions and should not be used in Layer 3
    
    noise_stats = f"Disorder={df['price_disorder_score'].mean():.3f}"
    tprint_success(f"✅ Advanced noise: {noise_stats}")
    tprint_success(f"✅ Added 1 advanced noise feature")
    
    # === Layer 1 Weight Features (4 features) ===
    if layer1_weight is not None:
        tprint_info("⚖️  Generating Layer 1 Weight Features...")
        layer1_w = pd.Series(layer1_weight, index=df.index)
        
        df['layer1_weight_momentum'] = layer1_w.pct_change()
        df['layer1_weight_volatility_adj'] = layer1_w / (df['volatility_1d'] + EPS)
        df['weight_confidence_score'] = 1 - np.abs(layer1_w - 1.0)
        df['weight_regime_indicator'] = (layer1_w > layer1_w.rolling(50).mean()).astype(int)
        
        weight_stats = f"mean={layer1_w.mean():.3f}, std={layer1_w.std():.3f}"
        tprint_success(f"✅ Layer 1 weights: {weight_stats}")
        tprint_success(f"✅ Added 4 Layer 1 weight features")
    else:
        tprint_info("📊 No Layer 1 weights provided, skipping weight features")
    
    # Final summary
    final_feature_count = len(df.columns)
    features_added = final_feature_count - initial_feature_count
    
    tprint_success(f"🎉 Feature Enhancement Complete!")
    tprint_success(f"📈 Features: {initial_feature_count} → {final_feature_count} (+{features_added})")
    
    # Feature category summary
    feature_summary = {
        'unified_price': 4,
        'adaptive_filter': 3,
        'noise_reduction': 3,
        'filter_consensus': 2,
        'advanced_noise': 4,
        'layer1_weights': 4 if layer1_weight is not None else 0
    }
    
    for category, count in feature_summary.items():
        if count > 0:
            tprint_info(f"   - {category}: {count} features")
    
    return df

def hierarchical_feature_filtering(X: pd.DataFrame, y: pd.Series, base_predictions: pd.Series, fast_mode: bool = False):
    """
    Multi-stage filtering to reduce CMI computations.
    
    Stages:
    1. Variance filter (remove low-variance features)
    2. Correlation filter (remove highly correlated features) - vectorized
    3. Fast MI proxy (top 50%) - skipped in fast mode
    4. Full CMI (top 20%) - skipped in fast mode
    """
    # Stage 1: Variance filter - vectorized
    var_mask = X.var() > 1e-6
    X_filtered = X.loc[:, var_mask]
    
    if fast_mode or len(X_filtered.columns) <= 50:
        # Skip expensive correlation filtering in fast mode or with few features
        return X_filtered

    # Stage 2: Correlation filter - vectorized for speed
    if len(X_filtered.columns) > 1:
        # Use numpy corrcoef for better performance than pandas corr()
        X_values = X_filtered.values.T  # Transpose for corrcoef (features x samples)
        corr_matrix = np.abs(np.corrcoef(X_values))
        np.fill_diagonal(corr_matrix, 0)  # Don't correlate with self

        # Find highly correlated pairs efficiently
        high_corr_mask = corr_matrix > 0.9
        high_corr_pairs = np.where(high_corr_mask)

        # Vectorized target correlation calculation using numpy
        y_values = y.values if hasattr(y, 'values') else y
        X_values = X_filtered.values
        target_corr = np.abs(np.array([np.corrcoef(X_values[:, i], y_values)[0, 1]
                                      for i in range(X_values.shape[1])]))

        # Use JIT-compiled feature selection for better performance
        features_to_keep_mask = select_features_by_correlation_jit(corr_matrix, target_corr, 0.9)
        features_to_keep_indices = np.where(features_to_keep_mask)[0]

        # Convert indices back to column names
        cols_to_keep = [X_filtered.columns[i] for i in features_to_keep_indices]
        X_filtered = X_filtered[cols_to_keep]

    return X_filtered
