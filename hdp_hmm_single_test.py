#!/usr/bin/env python3
"""
Single HDP-HMM test runner (isolation mode to avoid pyhsmm crashes)
Run as subprocess to prevent memory/semaphore leaks
"""
import sys
import os

# Suppress ALL logging output to keep stdout clean
# Only the final SUCCESS/ERROR line should go to stdout
os.environ['TPRINT_SILENT'] = '1'  # Tell tprint to be silent
os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress Python warnings

import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime, timedelta
import warnings
import logging

# Suppress all warnings and logging
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

# Don't redirect stderr - we need it for debugging subprocess issues

# Accept parameters from command line
if len(sys.argv) < 4:
    print("Usage: python hdp_hmm_single_test.py <alpha> <kappa> <gamma> [n_iterations] [--sensitivity_mode <mode>]")
    sys.exit(1)

alpha = float(sys.argv[1])
kappa = float(sys.argv[2])
gamma = float(sys.argv[3])
n_iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 30  # Default 30 if not specified

# Check for sensitivity mode
sensitivity_mode = "standard"
if len(sys.argv) > 5 and sys.argv[5] == "--sensitivity_mode" and len(sys.argv) > 6:
    sensitivity_mode = sys.argv[6]

# Don't print anything until the very end to keep stdout clean for parsing

try:
    from src.training.steps.market_analysis.hdp_hmm_clustering.hdp_hmm_clusterer import (
        HDPHMMClusterer, HDPHMMConfig, HMM_AVAILABLE
    )
    
    if not HMM_AVAILABLE:
        print(f"ERROR|{alpha}|{kappa}|{gamma}|HMM libraries not available", flush=True)
        sys.exit(1)
    
    # OPTIMIZED: Load pre-computed features from cache (MUCH faster!)
    # Run hdp_hmm_prepare_data.py once to create the cache file
    cache_file_npy = "hdp_hmm_features_cache.npy"
    
    if os.path.exists(cache_file_npy):
        # Load silently - no output during processing
        feature_array = np.load(cache_file_npy)
        
        # FIXED: Keep float64 for HDP-HMM (log-likelihoods prone to underflow with float32)
        # Cache may be float32, but convert back to float64 before HMM
        if feature_array.dtype == np.float64:
             feature_array = feature_array.astype(np.float32)        
            
        # ENHANCEMENT: Load price data for economic CV calculation
        price_cache_file = "hdp_hmm_price_cache.pkl"
        forward_returns = None
        timestamps = None
        if os.path.exists(price_cache_file):
            try:
                import pickle
                with open(price_cache_file, 'rb') as f:
                    price_data = pickle.load(f)
                
                # Calculate forward returns (1-bar, 5-bar, 10-bar)
                close_prices = price_data['close']
                timestamps = pd.DatetimeIndex(price_data['timestamp'])
                
                # Use 5-bar (5-hour) forward returns as primary economic metric
                returns_5h = np.zeros(len(close_prices))
                for i in range(len(close_prices) - 5):
                    returns_5h[i] = (close_prices[i+5] / close_prices[i]) - 1.0
                
                forward_returns = pd.Series(returns_5h, index=timestamps)
            except Exception:
                pass  # Silently continue if price data unavailable
    else:
        # Fallback: Load data the slow way if cache doesn't exist
        print("⚠️ Cache not found, loading data (slow)...", flush=True)
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
        
        # Generate features
        regime_integrator = RegimeFeatureIntegration()
        feature_chunks = []
        for i in range(0, len(df) - 50 + 1, 10):
            chunk = df.iloc[i:i+50]
            if len(chunk) >= 48:
                try:
                    regime_features = regime_integrator._generate_regime_features(chunk)
                    chunk_df = pd.DataFrame([regime_features]) if isinstance(regime_features, dict) else regime_features
                    feature_chunks.append(chunk_df)
                except:
                    continue
        
        feature_df = pd.concat(feature_chunks, ignore_index=True).fillna(0)
        
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object':
                try:
                    feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
                except:
                    feature_df[col] = pd.Categorical(feature_df[col]).codes
        feature_df = feature_df.fillna(0)
        
        # Two-scale normalization (REDUCED by 33%: 12h→8h, 48h→32h)
        feature_df_normalized = pd.DataFrame()
        for col in feature_df.columns:
            mean_8h = feature_df[col].rolling(8, min_periods=3).mean()
            std_8h = feature_df[col].rolling(8, min_periods=3).std()
            feature_df_normalized[f'{col}_short'] = (feature_df[col] - mean_8h) / (std_8h + 1e-8)
            
            mean_32h = feature_df[col].rolling(32, min_periods=8).mean()
            std_32h = feature_df[col].rolling(32, min_periods=8).std()
            feature_df_normalized[f'{col}_long'] = (feature_df[col] - mean_32h) / (std_32h + 1e-8)
        
        feature_df_normalized = feature_df_normalized.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Keep float64 for HDP-HMM stability (log-likelihoods need precision)
        feature_array = feature_df_normalized.values.astype(np.float64)
    
    # Run HDP-HMM with OPTIMIZED settings for speed
    # Burnin is ~15-20% of iterations
    n_burnin = max(2, int(n_iterations * 0.15))
    
    # FIXED: Separate random seeds for fair comparison
    # - Fixed seed (42) for HMM sampling ensures fair comparison across parameters
    # - Param-dependent seed for K-means allows initialization exploration
    import hashlib
    param_string = f"{alpha:.6f}_{kappa:.6f}_{gamma:.6f}"
    kmeans_seed_hash = int(hashlib.md5(param_string.encode()).hexdigest()[:8], 16)
    kmeans_seed = kmeans_seed_hash % (2**31)  # Param-dependent for K-means exploration
    
    # ALPHA-DEPENDENT K-MEANS INITIALIZATION (as requested by user)
    # Higher alpha → more regime diversity → initialize with more clusters
    # This gives alpha the expected theoretical effect on cluster count
    alpha_scaled = (alpha - 1.0) / (4.0 - 1.0)  # Scale [1.0, 4.0] to [0.0, 1.0]
    kmeans_init_clusters = int(3 + alpha_scaled * 7)  # Maps to [3, 10] clusters
    kmeans_init_clusters = max(3, min(10, kmeans_init_clusters))  # Clamp to [3, 10]
    
    config = HDPHMMConfig(
        alpha=alpha,
        kappa=kappa,
        gamma=gamma,
        n_iterations=n_iterations,           # Stage-dependent (30/100/200)
        n_burnin=n_burnin,                   # Proportional to iterations
        max_states=10,                       # Maximum number of states
        kmeans_n_clusters=kmeans_init_clusters,  # ALPHA-DEPENDENT: α controls initial cluster count
        pca_components=None,                 # OPTIMIZATION: PCA is now pre-computed
        covariance_type="diag",              # DIAGONAL covariance using SimpleDiagGaussian (~10x speedup!)
        use_gpu_acceleration=False,          # DISABLED to avoid GPU issues
        use_kmeans_warmstart=True,           # ENABLED - necessary for good initialization
        enable_advanced_diagnostics=False,   # DISABLED for speed
        convergence_check=True,              # FIXED: Enable early stopping to save time
        convergence_threshold=0.02,          # AGGRESSIVE: 0.02 (was 0.01) for faster convergence in grid search
        convergence_window=5,                # AGGRESSIVE: 5 (was 10) checks convergence 2x faster
        temporal_sensitivity_mode=sensitivity_mode,  # Sensitivity mode for temporal smoothness
        convergence_patience=3,              # AGGRESSIVE: 3 (was 5) stops sooner
        random_state=42,                     # FIXED: Same for all tests (fair comparison)
        kmeans_random_state=kmeans_seed,     # FIXED: Param-dependent for exploration
        show_progress=False
    )
    # NOTE: Alpha controls cluster count (3,5,7,10), κ controls persistence, γ controls distinctness
    
    # FIXED: Add minimal validation even when full validation is disabled
    # This catches corrupted cache or wrong shape before expensive HMM computation
    if feature_array.ndim != 2:
        print(f"ERROR|{alpha}|{kappa}|{gamma}|Invalid data shape: {feature_array.shape}", flush=True)
        sys.exit(1)
    
    if feature_array.shape[0] < 500:  # Minimum samples required
        print(f"ERROR|{alpha}|{kappa}|{gamma}|Insufficient samples: {feature_array.shape[0]}", flush=True)
        sys.exit(1)
    
    if np.any(np.isnan(feature_array)) or np.any(np.isinf(feature_array)):
        print(f"ERROR|{alpha}|{kappa}|{gamma}|Data contains NaN/Inf", flush=True)
        sys.exit(1)
    
    start_time = datetime.now()
    clusterer = HDPHMMClusterer(config)
    
    # OPTIMIZATION: Skip full validation during grid search (minimal checks done above)
    result = clusterer.fit_predict(feature_array, validate=False)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # ENHANCEMENT: Recalculate quality metrics with forward returns if available
    if forward_returns is not None and hasattr(clusterer, 'quality_assessor'):
        try:
            # Convert feature_array to DataFrame for quality assessor
            feature_df = pd.DataFrame(feature_array)
            
            # Recalculate quality assessment with economic data
            enhanced_quality = clusterer.quality_assessor.assess_hmm_regime_quality(
                regime_labels=result.cluster_labels,
                feature_data=feature_df,
                transition_matrix=result.transition_matrix if hasattr(result, 'transition_matrix') else None,
                hmm_model=None,  # Not needed for metric calculation
                forward_returns=forward_returns,
                timestamps=timestamps,
                timeframe="1h",
                min_regime_size=10,
                run_validators=False,  # Skip validators for speed
                temporal_sensitivity_mode=sensitivity_mode
            )
            # Update result with enhanced quality metrics
            result.quality_assessment = enhanced_quality.to_dict()
        except Exception:
            pass  # Continue with original quality assessment if enhancement fails
    
    # FIXED: Safe metric extraction with proper error handling
    def safe_metric(value, default=0.0, name="metric"):
        """Safely extract metric with validation."""
        if value is None:
            return default
        try:
            float_val = float(value)
            if np.isnan(float_val) or np.isinf(float_val):
                return default
            return float_val
        except (TypeError, ValueError):
            return default
    
    def safe_nested_get(d, *keys, default=0.0):
        """Safely get nested dictionary value."""
        try:
            current = d
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                    if current is None:
                        return default
                else:
                    return default
            return float(current) if current is not None else default
        except (TypeError, ValueError, AttributeError):
            return default
    
    @njit(fastmath=True, cache=True)
    def calculate_category_cv_ratios_numba(labels, feature_array, cat_indices_list):
        """
        Numba-accelerated CV ratio calculation.
        cat_indices_list is a list of lists, where each inner list has the
        column indices for a specific category.
        """
        # Numba doesn't support dictionaries, so we use a fixed-size array
        # 0=order_flow, 1=microstructure, 2=momentum, 3=volatility,
        # 4=volume, 5=trend, 6=temporal
        category_cvs = np.zeros(7, dtype=np.float64)
    
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        if n_clusters == 0:
            return category_cvs
    
        for cat_idx, cat_indices in enumerate(cat_indices_list):
            if len(cat_indices) == 0:
                continue
    
            # Use Numba-friendly loops to build the feature subset
            # This is faster inside Numba than advanced slicing
            cat_features = np.empty((feature_array.shape[0], len(cat_indices)), dtype=np.float64)
            for i in range(len(cat_indices)):
                cat_features[:, i] = feature_array[:, cat_indices[i]]
    
            # Calculate regime means and variances
            regime_means = np.empty((n_clusters, cat_features.shape[1]), dtype=np.float64)
            regime_vars = np.empty((n_clusters,), dtype=np.float64)
    
            for i, r in enumerate(unique_labels):
                mask = (labels == r)
                regime_data = cat_features[mask]
    
                if regime_data.shape[0] > 0:
                    regime_means[i] = np.mean(regime_data, axis=0)
                    regime_vars[i] = np.mean(np.var(regime_data, axis=0))
                else:
                    regime_means[i] = np.zeros(cat_features.shape[1], dtype=np.float64)
                    regime_vars[i] = 0.0
    
            # Calculate between-regime and within-regime CV
            between_var = np.mean(np.var(regime_means, axis=0))
            within_var = np.mean(regime_vars)
    
            if within_var > 1e-9:
                category_cvs[cat_idx] = between_var / within_var
            else:
                category_cvs[cat_idx] = 0.0
    
        return category_cvs
        
    # --- THIS IS THE ORIGINAL (NON-NUMBA) FUNCTION ---
    def calculate_category_cv_ratios(labels, features, feature_array):
        """Calculates CV ratios by preparing indices and calling the Numba function."""
        try:
            import pickle
            try:
                with open('hdp_hmm_features_cache.pkl', 'rb') as f:
                    feature_df = pickle.load(f)
                    feature_names = list(feature_df.columns)
            except:
                feature_names = [f"feat_{i}" for i in range(feature_array.shape[1])]
    
            categories = {
                'order_flow': ['volume_momentum', 'volume_clustering', 'volume_roc'],
                'microstructure': ['price_zscore', 'mean_reversion', 'volume_clustering'],
                'momentum': ['momentum_', 'roc_', '_acceleration'],
                'volatility': ['volatility', 'lagged_range', 'range_ratio'],
                'volume': ['volume_ratio', 'lagged_volume'],
                'trend': ['price_to_ma', 'trend_strength', 'temporal_price'],
                'temporal': ['regime_duration', 'lagged_']
            }
    
            category_names = ['order_flow', 'microstructure', 'momentum', 'volatility', 'volume', 'trend', 'temporal']
            cat_indices_list = []
    
            for cat_name in category_names:
                patterns = categories[cat_name]
                cat_indices = []
                for idx, fname in enumerate(feature_names):
                    if any(pattern in fname for pattern in patterns):
                        cat_indices.append(idx)
                # Numba needs a list of np.arrays for typed lists
                cat_indices_list.append(np.array(cat_indices, dtype=np.int32))
    
            # --- CALL THE NEW NUMBA FUNCTION ---
            cv_values = calculate_category_cv_ratios_numba(labels, feature_array, cat_indices_list)
    
            # Map array back to dictionary
            return {name: val for name, val in zip(category_names, cv_values)}
    
        except Exception as e:
            return {cat: 0.0 for cat in ['order_flow', 'microstructure', 'momentum', 
                                          'volatility', 'volume', 'trend', 'temporal']}
    
    # Output results in parseable format
    if result.success:
        # Extract quality metrics from quality_assessment dictionary
        qa = result.quality_assessment or {}
        
        # FIXED: Safe metric extraction with proper handling
        temporal = safe_metric(qa.get('temporal_smoothness'), 0.0, 'temporal_smoothness')
        balance = safe_metric(qa.get('balance_score'), 0.0, 'balance_score')
        between_cv = safe_metric(qa.get('between_regime_cv'), 0.0, 'between_regime_cv')
        
        # FIXED: Epsilon-safe division (within_cv used in CV ratio calculation)
        within_cv_raw = qa.get('within_regime_cv')
        if within_cv_raw is None or within_cv_raw == 0:
            within_cv = 1.0  # Safe default for division
        else:
            within_cv = safe_metric(within_cv_raw, 1.0, 'within_regime_cv')
        
        # FIXED: Safe nested dictionary access for economic CV
        # NOTE: The actual key is 'economic_cv_ratio_mean_return' (flat), not nested structure
        economic_cv = safe_nested_get(
            qa, 'economic_cv_metrics', 'economic_cv_ratio_mean_return',
            default=0.0
        )
        if economic_cv == 0.0:
            # Fallback: calculate ratio from between/within if available
            between_cv = safe_nested_get(qa, 'economic_cv_metrics', 'economic_between_cv_mean_return', default=0.0)
            within_cv = safe_nested_get(qa, 'economic_cv_metrics', 'economic_avg_within_cv_fwd_return', default=1.0)
            if within_cv > 1e-9:
                economic_cv = between_cv / within_cv
        
        # Silhouette score - use from result or quality_assessment
        silhouette = safe_metric(result.silhouette_score, 0.0, 'silhouette_score')
        if silhouette == 0.0:
            silhouette = safe_metric(qa.get('silhouette_score'), 0.0, 'silhouette_score')
        
        # FIXED: Extract convergence information if available
        converged = False
        convergence_iteration = n_iterations
        if hasattr(clusterer, 'convergence_history') and clusterer.convergence_history:
            conv_info = clusterer.convergence_history
            converged = conv_info.get('converged', False) if isinstance(conv_info, dict) else False
            if isinstance(conv_info, dict):
                # FIX: Use 'or' instead of get() default to handle None values correctly
                convergence_iteration = conv_info.get('convergence_iteration') or n_iterations
        
        # Calculate CV ratio (between / within)
        cv_ratio = between_cv / (within_cv + 1e-9) if within_cv > 0 else 0.0
        
        # Calculate per-category CV ratios
        cluster_labels = getattr(result, 'cluster_labels', None)
        if cluster_labels is not None:
            category_cvs = calculate_category_cv_ratios(cluster_labels, None, feature_array)
        else:
            category_cvs = {cat: 0.0 for cat in ['order_flow', 'microstructure', 'momentum', 
                                                  'volatility', 'volume', 'trend', 'temporal']}
        
        # Print result to stdout (subprocess will capture this)
        # Format: SUCCESS|α|κ|γ|clusters|silhouette|temporal|balance|between_cv|within_cv|cv_ratio|economic_cv|elapsed|converged|conv_iter|cat_cvs...
        cat_cv_str = '|'.join([f"{category_cvs.get(cat, 0.0):.4f}" 
                               for cat in ['order_flow', 'microstructure', 'momentum', 
                                          'volatility', 'volume', 'trend', 'temporal']])
        
        print(f"SUCCESS|{alpha}|{kappa}|{gamma}|{result.n_clusters}|{silhouette}|"
              f"{temporal}|{balance}|{between_cv}|{within_cv}|{cv_ratio}|{economic_cv}|{elapsed}|"
              f"{int(converged)}|{convergence_iteration}|{cat_cv_str}", 
              flush=True)
    else:
        print(f"FAILED|{alpha}|{kappa}|{gamma}|{result.error_message}", 
              flush=True)
        
except Exception as e:
    print(f"ERROR|{alpha}|{kappa}|{gamma}|{str(e)}", 
          flush=True)
    sys.exit(1)

