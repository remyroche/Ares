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

# Import optimized functions
try:
    from src.training.steps.labeling.optimized_layer2_functions import (
        _vectorized_rolling_features,
        vectorized_feature_selection
    )
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False
    print("⚠️ Optimized Layer 2 functions not available, falling back to standard implementation")

try:
    from src.training.steps.labeling.layer3.optimized_utils import numba_feature_target_correlation
    OPTIMIZED_UTILS_AVAILABLE = True
except ImportError:
    OPTIMIZED_UTILS_AVAILABLE = False

from src.training.steps.labeling.feature_engineering_utils import apply_layer2_price_processing

# Import RuleFit for interaction features
try:
    from src.training.steps.labeling.generate_interaction_features_et_rulefit import (
        RuleFitTransformer, LeafGateConfig
    )
    # add_enhanced_gates was moved to layer2_5_chaser
    from src.training.steps.labeling.layer2_5_chaser import add_enhanced_gates
    RULEFIT_AVAILABLE = True
except ImportError:
    RULEFIT_AVAILABLE = False
    print("⚠️ RuleFit not available, interaction features will be skipped")


# Import Unified Cache
try:
    from src.training.steps.labeling.layer3_feature_cache import (
        save_layer3_features_to_cache,
        load_layer3_features_from_cache
    )
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    print("⚠️ Layer3 Feature Cache not available, falling back to local implementation")

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

def downcast_float(df: pd.DataFrame) -> pd.DataFrame:
    """Safely downcast float64 columns to float32 to save memory and speed up computation."""
    cols = df.select_dtypes(include=['float64']).columns
    if len(cols) > 0:
        df[cols] = df[cols].astype(np.float32)
    return df

@njit(fastmath=True)
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
    # Optimize: Don't create full mask matrix, just iterate

    for i in range(n_features):
        if not features_to_keep[i]:
            continue

        for j in range(i + 1, n_features):  # Only check upper triangle
            if not features_to_keep[j]:
                continue

            if corr_matrix[i, j] > threshold:
                # Keep the one with higher correlation to target
                if target_corr[i] < target_corr[j]:
                    features_to_keep[i] = False
                    break # i is dropped, stop checking j for this i
                else:
                    features_to_keep[j] = False

    return features_to_keep

def enhance_layer3_features_optimized(df, market_data, layer1_weight, layer0_params=None, fast_mode=False):
    """
    Add optimized Layer 0 feature combinations, noise features, and Layer 1 weight integration.
    
    Uses intelligent combinations of Layer 0 filtering methods rather than individual features.
    Now optimized with downcasting and Numba.

    Args:
        fast_mode: Skip expensive computations for faster execution
    """
    # Downcast input to float32
    df = downcast_float(df)
    if market_data is not None:
        market_data = downcast_float(market_data.copy())

    if fast_mode:
        tprint_info("⚡ Fast mode: minimal Layer 0 feature integration")
        # Add just basic momentum features
        if 'close' in market_data.columns:
            close = market_data['close'].values
            df['fast_momentum'] = np.concatenate([np.array([np.nan]), close[1:] / close[:-1] - 1]).astype(np.float32)

            # Fast rolling std using Numba if available
            if OPTIMIZED_AVAILABLE:
                # _vectorized_rolling_features returns [mean, std, range] for each window
                # We want window 20 std, so index 1
                features_20 = _vectorized_rolling_features(close.astype(np.float32), np.array([20], dtype=np.int32))
                df['fast_volatility'] = features_20[:, 1].astype(np.float32)
            else:
                df['fast_volatility'] = market_data['close'].rolling(20).std().astype(np.float32)
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
        enabled_features.append('Adaptive Kalman')
    if layer0_params.get('robust_vwap_enabled', False):
        enabled_features.append('Robust VWAP')
    if layer0_params.get('hampel_filter_enabled', False):
        enabled_features.append('Hampel Filter')
    if layer0_params.get('savgol_filter_enabled', False):
        enabled_features.append('Savitzky-Golay')
    
    tprint_info(f"🔧 Layer 0 enabled features: {enabled_features}")
    
    initial_feature_count = len(df.columns)
    
    # === 0. Anti-Explosion Feature Integration (Layer 2 Processing) ===
    if market_data is not None and 'close' in market_data.columns:
        try:
            # Generate price features on market data
            price_feats = apply_layer2_price_processing(market_data, price_col='close', enable_price_features=True)

            # Select new features (excluding OHLCV if present)
            new_cols = [c for c in price_feats.columns if c not in market_data.columns and c not in df.columns]

            # Reindex to align with df (df might be subset of market_data, e.g. events)
            # Use merge_asof or reindex. market_data usually covers full range.
            # Assuming df index is subset of market_data index.

            aligned_feats = price_feats[new_cols].reindex(df.index).fillna(0.0)

            # Add to df
            df = pd.concat([df, aligned_feats], axis=1)
            tprint_success(f"✅ Added {len(new_cols)} Anti-Explosion features from Layer 2 processing")

        except Exception as e:
            tprint_warning(f"⚠️ Anti-Explosion feature integration failed: {e}")

    # === Optimized Layer 0 Feature Combinations ===
    
    # 1. Unified Filtered Price
    tprint_info("📈 Generating Unified Filtered Price...")

    unified_price = None

    # Try to load from unified cache if available
    # Note: Layer3FeatureCache requires specific metadata (symbol, exchange, etc) which we might not have here
    # If we assume standard defaults or if layer0_params has them
    cache_used = False

    try:
        from src.training.steps.labeling.unified_price_layer2 import generate_unified_layer2_price
        unified_price = generate_unified_layer2_price(market_data, layer0_params)
        unified_price = unified_price.reindex(df.index).fillna(method='ffill').astype(np.float32)
        
        # Core unified price features (4 features)
        # Use vectorized operations
        unified_vals = unified_price.values
        df['unified_price_momentum'] = np.concatenate([np.array([np.nan]), unified_vals[1:] / unified_vals[:-1] - 1]).astype(np.float32)

        if OPTIMIZED_AVAILABLE:
            # Calculate rolling mean (20) and std (20)
            feat_20 = _vectorized_rolling_features(unified_vals.astype(np.float32), np.array([20], dtype=np.int32))
            # feat_20 col 0 = mean, col 1 = std
            df['unified_price_strength'] = (unified_vals > feat_20[:, 0]).astype(int)

            # Using market_data['volatility_1d'] which should be present
            vol_1d = market_data['volatility_1d'].values
            df['unified_volatility_adj'] = (feat_20[:, 1] / (vol_1d + EPS)).astype(np.float32)
        else:
            df['unified_price_strength'] = (unified_price > unified_price.rolling(20).mean()).astype(int)
            df['unified_volatility_adj'] = unified_price.rolling(20).std() / market_data['volatility_1d']

        close_vals = market_data['close'].values.astype(np.float32)
        df['unified_regime_confidence'] = (1 - np.abs(unified_vals - close_vals) / close_vals).astype(np.float32)
        
        unified_stats = f"mean={unified_price.mean():.4f}, std={unified_price.std():.4f}"
        tprint_success(f"✅ Unified price: {unified_stats}")
        tprint_success(f"✅ Added 4 unified price features")
        
    except Exception as e:
        tprint_warning(f"⚠️ Unified price features failed: {e}")
    
    # 2. Adaptive Filtering Score
    tprint_info("🔄 Generating Adaptive Filtering Features...")
    adaptive_features = []
    
    if layer0_params.get('adaptive_kalman_enabled', False):
        try:
            from src.training.steps.labeling.unified_price_layer2 import generate_adaptive_kalman_price
            adaptive_price = generate_adaptive_kalman_price(
                market_data,
                base_Q=layer0_params.get('kalman_Q', 1e-4),
                base_R=layer0_params.get('kalman_R', 0.01),
                noise_window=layer0_params.get('adaptive_noise_window', 50),
                adaptation_rate=layer0_params.get('adaptive_adaptation_rate', 0.1)
            )
            adaptive_features.append(adaptive_price.reindex(df.index).fillna(method='ffill').astype(np.float32))
            tprint_success("✅ Generated adaptive Kalman price")
        except Exception as e:
            tprint_warning(f"⚠️ Adaptive Kalman failed: {e}")
    
    if layer0_params.get('robust_vwap_enabled', False) and 'volume' in market_data.columns:
        try:
            from src.training.steps.labeling.unified_price_layer2 import generate_robust_vwap_price
            robust_vwap = generate_robust_vwap_price(
                market_data,
                base_lookback=layer0_params.get('vwap_lookback', 50),
                min_lookback=layer0_params.get('robust_min_lookback', 20),
                max_lookback=layer0_params.get('robust_max_lookback', 200),
                volatility_window=layer0_params.get('robust_volatility_window', 20)
            )
            robust_vwap_series = robust_vwap.reindex(df.index).fillna(method='ffill').astype(np.float32)
            adaptive_features.append(robust_vwap_series)
            df['vwap'] = robust_vwap_series
            if market_data is not None:
                market_data['vwap'] = robust_vwap.reindex(market_data.index).fillna(method='ffill').astype(np.float32)
            tprint_success("✅ Generated robust VWAP price")
        except Exception as e:
            tprint_warning(f"⚠️ Robust VWAP failed: {e}")
    
    # Combine adaptive features
    if len(adaptive_features) > 1:
        adaptive_combined = pd.concat(adaptive_features, axis=1).mean(axis=1).astype(np.float32)
        tprint_info(f"🔄 Combined {len(adaptive_features)} adaptive methods")
    elif len(adaptive_features) == 1:
        adaptive_combined = adaptive_features[0]
        tprint_info("🔄 Using single adaptive method")
    else:
        adaptive_combined = market_data['close'].reindex(df.index).astype(np.float32)
        tprint_warning("⚠️ No adaptive features available, using close price")
    
    # Adaptive filtering features (3 features)
    adapt_vals = adaptive_combined.values
    df['adaptive_filter_momentum'] = np.concatenate([np.array([np.nan]), adapt_vals[1:] / adapt_vals[:-1] - 1]).astype(np.float32)

    # Safe alignment
    close_vals = market_data['close'].reindex(df.index).values.astype(np.float32)
    df['adaptive_filter_distance'] = ((close_vals - adapt_vals) / (close_vals + EPS)).astype(np.float32)

    if OPTIMIZED_AVAILABLE:
        # Rolling mean 50
        feat_50 = _vectorized_rolling_features(adapt_vals.astype(np.float32), np.array([50], dtype=np.int32))
        df['adaptive_filter_regime'] = (adapt_vals > feat_50[:, 0]).astype(int)
    else:
        df['adaptive_filter_regime'] = (adaptive_combined > adaptive_combined.rolling(50).mean()).astype(int)
    
    tprint_success(f"✅ Added 3 adaptive filtering features")
    
    # 3. Noise Reduction Score
    tprint_info("🔇 Generating Noise Reduction Features...")
    noise_reduction_features = []
    
    if layer0_params.get('hampel_filter_enabled', False):
        try:
            from src.training.steps.labeling.unified_price_layer2 import apply_hampel_filter
            hampel_price = apply_hampel_filter(
                market_data['close'],
                window=layer0_params.get('hampel_window', 5),
                threshold=layer0_params.get('hampel_threshold', 3.0)
            )
            noise_reduction_features.append(hampel_price.reindex(df.index).astype(np.float32))
            tprint_success("✅ Generated Hampel filtered price")
        except Exception as e:
            tprint_warning(f"⚠️ Hampel filter failed: {e}")
    
    if layer0_params.get('savgol_filter_enabled', False):
        try:
            from src.training.steps.labeling.unified_price_layer2 import apply_savgol_filter
            savgol_price = apply_savgol_filter(
                market_data['close'],
                window_length=layer0_params.get('savgol_window', 21),
                poly_order=layer0_params.get('savgol_order', 3)
            )
            noise_reduction_features.append(savgol_price.reindex(df.index).astype(np.float32))
            tprint_success("✅ Generated Savitzky-Golay filtered price")
        except Exception as e:
            tprint_warning(f"⚠️ Savitzky-Golay filter failed: {e}")
    
    # Combine noise reduction features
    if len(noise_reduction_features) > 1:
        noise_combined = pd.concat(noise_reduction_features, axis=1).mean(axis=1).astype(np.float32)
        tprint_info(f"🔇 Combined {len(noise_reduction_features)} noise reduction methods")
    elif len(noise_reduction_features) == 1:
        noise_combined = noise_reduction_features[0]
        tprint_info("🔇 Using single noise reduction method")
    else:
        noise_combined = market_data['close'].reindex(df.index).astype(np.float32)
        tprint_warning("⚠️ No noise reduction features available, using close price")
    
    # Noise reduction features (3 features)
    noise_vals = noise_combined.values
    df['noise_reduction_momentum'] = np.concatenate([np.array([np.nan]), noise_vals[1:] / noise_vals[:-1] - 1]).astype(np.float32)

    # Safe alignment using already reindexed close_vals
    df['noise_reduction_smoothness'] = (1 - np.abs(noise_vals - close_vals) / (close_vals + EPS)).astype(np.float32)
    df['noise_reduction_stability'] = (np.abs(noise_vals - close_vals) < 0.01).astype(int)
    
    tprint_success(f"✅ Added 3 noise reduction features")
    
    # 4. Filter Consensus Score
    tprint_info("🤝 Calculating Filter Consensus...")
    all_filtered_prices = []
    if unified_price is not None:
        all_filtered_prices.append(unified_price)
    else:
        unified_price = market_data['close'].reindex(df.index).astype(np.float32)
        all_filtered_prices.append(unified_price)

    all_filtered_prices.extend(adaptive_features)
    all_filtered_prices.extend(noise_reduction_features)
    
    if len(all_filtered_prices) >= 2:
        filter_matrix = pd.concat(all_filtered_prices, axis=1)
        filter_consensus = filter_matrix.std(axis=1).astype(np.float32)
        df['filter_consensus_score'] = (1 - (filter_consensus / market_data['close'])).astype(np.float32)

    if len(all_filtered_prices) >= 2:
        filter_matrix = pd.concat(all_filtered_prices, axis=1)
        filter_consensus = filter_matrix.std(axis=1).astype(np.float32)
        # Safe alignment
        close_vals_idx = market_data['close'].reindex(df.index).values.astype(np.float32)
        df['filter_consensus_score'] = (1 - (filter_consensus / (close_vals_idx + EPS))).astype(np.float32)
        
        consensus_stats = f"mean={df['filter_consensus_score'].mean():.3f}"
        tprint_success(f"✅ Filter consensus: {consensus_stats}")
        tprint_success(f"✅ Added 1 consensus feature")
    else:
        df['filter_consensus_score'] = 1.0
    
    # === Advanced Noise Features ===
    tprint_info("📊 Generating Advanced Noise Features...")
    
    # Price disorder score
    # Ensure market_data is aligned or compute on reindexed series
    close_series = market_data['close'].reindex(df.index)
    
    if OPTIMIZED_AVAILABLE:
        # std 20 and std 100
        close_vals_d = close_series.values.astype(np.float32)
        feat_noise = _vectorized_rolling_features(close_vals_d, np.array([20, 100], dtype=np.int32))
        # 20: col 1, 100: col 4
        df['price_disorder_score'] = (feat_noise[:, 1] / (feat_noise[:, 4] + EPS)).astype(np.float32)
    else:
        df['price_disorder_score'] = (close_series.rolling(20).std() / (close_series.rolling(100).std() + EPS)).astype(np.float32)
    
    noise_stats = f"Disorder={df['price_disorder_score'].mean():.3f}"
    tprint_success(f"✅ Advanced noise: {noise_stats}")
    tprint_success(f"✅ Added 1 advanced noise feature")
    
    # === Layer 1 Weight Features ===
    if layer1_weight is not None:
        tprint_info("⚖️  Generating Layer 1 Weight Features...")
        layer1_w = pd.Series(layer1_weight, index=df.index).astype(np.float32)
        w_vals = layer1_w.values

        df['layer1_weight_momentum'] = np.concatenate([np.array([np.nan]), w_vals[1:] / (w_vals[:-1] + EPS) - 1]).astype(np.float32)
        
        # Use volatility from market_data if not in df
        if 'volatility_1d' in df.columns:
            vol_vals = df['volatility_1d'].values
        elif 'volatility_1d' in market_data.columns:
            vol_vals = market_data['volatility_1d'].values
        else:
            vol_vals = np.ones_like(w_vals) * 0.01 # Fallback

        # Ensure vol_vals is aligned if it came from market_data directly
        if len(vol_vals) != len(df):
             # This means we pulled it from unaligned market_data
             if 'volatility_1d' in market_data.columns:
                 vol_vals = market_data['volatility_1d'].reindex(df.index).values
             else:
                 vol_vals = np.ones_like(w_vals) * 0.01

        df['layer1_weight_volatility_adj'] = (w_vals / (vol_vals + EPS)).astype(np.float32)
        df['weight_confidence_score'] = (1 - np.abs(w_vals - 1.0)).astype(np.float32)

        if OPTIMIZED_AVAILABLE:
            feat_w_50 = _vectorized_rolling_features(w_vals.astype(np.float32), np.array([50], dtype=np.int32))
            df['weight_regime_indicator'] = (w_vals > feat_w_50[:, 0]).astype(int)
        else:
            df['weight_regime_indicator'] = (layer1_w > layer1_w.rolling(50).mean()).astype(int)
        
        weight_stats = f"mean={layer1_w.mean():.3f}, std={layer1_w.std():.3f}"
        tprint_success(f"✅ Layer 1 weights: {weight_stats}")
        tprint_success(f"✅ Added 4 Layer 1 weight features")
    else:
        tprint_info("📊 No Layer 1 weights provided, skipping weight features")
    
    final_feature_count = len(df.columns)
    features_added = final_feature_count - initial_feature_count
    
    tprint_success(f"🎉 Feature Enhancement Complete!")
    tprint_success(f"📈 Features: {initial_feature_count} → {final_feature_count} (+{features_added})")
    
    return downcast_float(df)

def hierarchical_feature_filtering(X: pd.DataFrame, y: pd.Series, base_avg: pd.Series, fast_mode: bool = False) -> pd.DataFrame:
    """
    Apply hierarchical feature selection:
    1. Low variance filter
    2. High correlation filter (redundancy reduction) - Using numpy/float32 optimization
    3. Conditional importance filter (CMI proxy)
    """
    tprint_info(f"📊 Running Hierarchical Filtering ({len(X.columns)} features)...")
    
    # Ensure float32 for speed
    X = downcast_float(X)

    # 1. Variance Filter
    variances = X.var()
    low_var_cols = variances[variances < 1e-6].index
    if len(low_var_cols) > 0:
        X = X.drop(columns=low_var_cols)
        tprint_info(f"   📉 Variance: Removed {len(low_var_cols)} constant/low-variance features.")
        
    if X.empty: return X
    
    # 2. Correlation Filter (Redundancy)
    if not fast_mode and len(X.columns) > 1:
        # Optimized correlation calculation
        X_vals = X.values.T  # corrcoef expects rows as variables

        # Handle NaNs: fill with 0 (assuming standardized or robust) or mean
        if np.isnan(X_vals).any():
             X_vals = np.nan_to_num(X_vals, nan=0.0)

        # Compute correlation matrix
        # For very large N, we might want to subsample?
        # But here N is usually < 50k, F < 500, so 500x500 matrix is small.
        try:
            corr_matrix = np.abs(np.corrcoef(X_vals))
        except Exception:
            # Fallback for weird edge cases
            corr_matrix = X.corr().abs().values

        # Calculate target correlation (Vectorized)
        y_val = y.values
        # Handle NaNs in y
        mask_y = ~np.isnan(y_val)
        
        if mask_y.all():
            X_clean_T = X_vals
            y_clean = y_val
        else:
            X_clean_T = X_vals[:, mask_y]
            y_clean = y_val[mask_y]

        # Standardize for correlation calculation
        if X_clean_T.shape[1] > 0:
            if OPTIMIZED_UTILS_AVAILABLE:
                 # Transpose back to (N, F) for numba function which expects that
                 target_corr = numba_feature_target_correlation(X_clean_T.T, y_clean)
                 target_corr = np.nan_to_num(target_corr, nan=0.0)
            else:
                y_mean = np.mean(y_clean)
                y_std = np.std(y_clean) + EPS
                y_norm = (y_clean - y_mean) / y_std

                X_mean = np.mean(X_clean_T, axis=1, keepdims=True)
                X_std = np.std(X_clean_T, axis=1, keepdims=True) + EPS
                X_norm = (X_clean_T - X_mean) / X_std

                # Correlation = dot(X_norm, y_norm) / N
                target_corr = np.abs(np.dot(X_norm, y_norm) / X_clean_T.shape[1])
                target_corr = np.nan_to_num(target_corr, nan=0.0)
        else:
            target_corr = np.zeros(X_clean_T.shape[0])

        # Use updated JIT function (faster loop)
        keep_mask = select_features_by_correlation_jit(corr_matrix, target_corr, threshold=0.95)
        keep_cols = X.columns[keep_mask]
        X = X[keep_cols]
        tprint_info(f"   📉 Redundancy: Reduced to {len(X.columns)} uncorrelated features.")
        
    # 3. Conditional Importance (CMI Proxy)
    if not fast_mode and len(X.columns) > 20:
        from .utils import fast_cmi_proxy
        X, _ = fast_cmi_proxy(X, (y > 0.5).astype(int), base_avg, top_percentile=0.7)
        tprint_info(f"   📉 CMI: Selected top {len(X.columns)} features.")
        
    return X

def apply_layer3_feature_selection(X: pd.DataFrame, y: pd.Series, base_predictions: pd.DataFrame, fast_mode: bool = False) -> pd.DataFrame:
    """
    [DE PRADO 2026] Complete Layer 3 Feature Selection Suite.
    1. SSFI Pruning (for disagreement features)
    2. Hierarchical Filtering (Variance + Correlation)
    """
    tprint_info(f"🔍 Layer 3: Running Feature Selection ({len(X.columns)} features)...")
    initial_cols = list(X.columns)
    
    # 1. SSFI Pruning
    disagreement_cols = [c for c in X.columns if c.endswith('_disagreement')]
    if disagreement_cols:
        from src.feature_generation.categories.layer3_specific_features import _apply_ssfi_pruning
        pruned_ssfi = _apply_ssfi_pruning(X, y, disagreement_cols)
        X = X.drop(columns=pruned_ssfi)
        tprint_info(f"   📉 SSFI: Removed {len(pruned_ssfi)} uninformative disagreement features.")
    
    # 1.5. RuleFit Interaction Features (NEW - before hierarchical filtering)
    if RULEFIT_AVAILABLE and not fast_mode:
        try:
            tprint_info("   🧬 Generating RuleFit Interaction Features...")
            
            # Add gates if OHLCV available
            ohlcv_cols = ['close', 'high', 'low', 'volume']
            gate_cols = [c for c in X.columns if c.startswith('g_')]
            
            if not gate_cols and all(c in X.columns for c in ohlcv_cols):
                X_with_gates = add_enhanced_gates(X)
                gate_cols = [c for c in X_with_gates.columns if c.startswith('g_')]
                for c in gate_cols:
                    X[c] = X_with_gates[c]
            
            if gate_cols:
                base_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                base_cols = [c for c in base_cols if c not in gate_cols]
                
                # Use smaller config for Layer 3 (it has fewer samples typically)
                rf_cfg = LeafGateConfig(
                    n_estimators=100,
                    max_depth=2,
                    n_stability_runs=20,  # Faster for Layer 3
                    stability_threshold=0.5
                )
                
                rulefit = RuleFitTransformer(
                    base_cols=base_cols,
                    gate_cols=gate_cols,
                    config=rf_cfg,
                    verbose=False
                )
                
                X_interactions = rulefit.fit_transform(X, y, class_weight="balanced")
                
                if not X_interactions.empty:
                    X = pd.concat([X, X_interactions], axis=1)
                    tprint_success(f"   ✅ Added {X_interactions.shape[1]} RuleFit interaction features")
            else:
                tprint_info("   ⚠️ No gate columns for RuleFit, skipping.")
                
        except Exception as e:
            tprint_warning(f"   ⚠️ RuleFit generation failed: {e}")
        
    # 2. Hierarchical Filtering
    # We use the mean of base predictions as a proxy for 'base_predictions' series
    base_avg = base_predictions.mean(axis=1) if isinstance(base_predictions, pd.DataFrame) else base_predictions
    X = hierarchical_feature_filtering(X, y, base_avg, fast_mode)

    
    final_cols = list(X.columns)
    removed = set(initial_cols) - set(final_cols)
    tprint_success(f"✅ Feature Selection Complete: {len(initial_cols)} -> {len(final_cols)} (-{len(removed)})")
    
    return X
