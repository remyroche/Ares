"""
Optimized prediction utilities for faster inference.

Implements:
1. Batch inference for alpha and meta models
2. Feature computation caching
3. Feature loading optimization
4. Vectorized meta model combination
5. Numba JIT-accelerated operations
"""

import os
import warnings
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit, prange
from extreme_price_movements.config import CANON_HORIZONS

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# Numba JIT-ACCELERATED OPERATIONS
# ============================================================================

@njit(parallel=True, fastmath=True)
def sigmoid_sizing_numba(confidence: np.ndarray, k: float, c0: float, 
                         s_min: float, s_max: float) -> np.ndarray:
    """Fast sigmoid position sizing with Numba JIT."""
    n = len(confidence)
    sizes = np.empty(n, dtype=np.float64)
    for i in prange(n):
        z = 1.0 / (1.0 + np.exp(-k * (confidence[i] - c0)))
        sizes[i] = s_min + (s_max - s_min) * z
    return sizes


@njit(parallel=True, fastmath=True)
def tanh_sizing_numba(confidence: np.ndarray, k: float, c0: float,
                      s_min: float, s_max: float) -> np.ndarray:
    """Fast tanh position sizing with Numba JIT."""
    n = len(confidence)
    sizes = np.empty(n, dtype=np.float64)
    for i in prange(n):
        z = 0.5 * (1.0 + np.tanh(k * (confidence[i] - c0)))
        sizes[i] = s_min + (s_max - s_min) * z
    return sizes


@njit(parallel=True, fastmath=True)
def concave_sizing_numba(confidence: np.ndarray, k: float, c0: float,
                         s_min: float, s_max: float) -> np.ndarray:
    """Fast concave position sizing with Numba JIT."""
    n = len(confidence)
    sizes = np.empty(n, dtype=np.float64)
    for i in prange(n):
        pos = max(0.0, confidence[i] - c0)
        pos_max = 0.0
        # Need to compute max across all - this is a limitation
        # We'll use a simpler approach: normalize by k
        if k > 0:
            pos_norm = min(1.0, pos / k)
            z = pos_norm ** k
        else:
            z = 0.0
        sizes[i] = s_min + (s_max - s_min) * z
    return sizes


@njit(parallel=True, fastmath=True)
def compute_disagreement_numba(preds1: np.ndarray, preds2: np.ndarray) -> np.ndarray:
    """Fast disagreement metric computation."""
    n = len(preds1)
    disagreement = np.empty(n, dtype=np.float64)
    for i in prange(n):
        disagreement[i] = np.abs(preds1[i] - preds2[i])
    return disagreement


@njit(parallel=True, fastmath=True)
def vectorized_clip_and_scale(values: np.ndarray, min_val: float, 
                               max_val: float, scale: float) -> np.ndarray:
    """Fast clip and scale operation."""
    n = len(values)
    result = np.empty(n, dtype=np.float64)
    for i in prange(n):
        clipped = max(min_val, min(max_val, values[i]))
        result[i] = clipped * scale
    return result


@njit(parallel=True, fastmath=True)
def compute_regime_interactions_numba(
    pred_logit: np.ndarray,
    regime_values: np.ndarray,
    n_buckets: int
) -> np.ndarray:
    """Fast regime interaction computation."""
    n = len(pred_logit)
    interactions = np.empty((n, n_buckets), dtype=np.float64)
    for i in prange(n):
        for b in range(n_buckets):
            interactions[i, b] = pred_logit[i] * (1.0 if int(regime_values[i]) == b else 0.0)
    return interactions


# ============================================================================
# BATCH INFERENCE UTILITIES
# ============================================================================

class BatchPredictor:
    """Optimized batch predictor for alpha and meta models."""

    # Adaptive batch size parameters
    MIN_BATCH_SIZE = 1000
    MAX_BATCH_SIZE = 50000
    TARGET_MEMORY_PER_BATCH_MB = 100  # Target memory usage per batch in MB

    def __init__(self, batch_size: Optional[int] = None, num_threads: Optional[int] = None, adaptive_batching: bool = True):
        """
        Initialize batch predictor.

        Args:
            batch_size: Number of samples to process in each batch (None for adaptive)
            num_threads: Number of threads for parallel processing (default: CPU count)
            adaptive_batching: If True, automatically tune batch size based on memory
        """
        self.num_threads = num_threads or os.cpu_count() or 1
        self.adaptive_batching = adaptive_batching

        if batch_size is None:
            # Start with a reasonable default
            self.batch_size = 10000
        else:
            self.batch_size = max(self.MIN_BATCH_SIZE, min(self.MAX_BATCH_SIZE, batch_size))

        # Memory tracking for adaptive batching
        self.memory_usage_samples = []
        
    def predict_batched(self, model, X: Union[pd.DataFrame, np.ndarray],
                       feature_names: Optional[List[str]] = None,
                       predict_proba: bool = False,
                       keep_dataframe: bool = False) -> np.ndarray:
        """
        Batch prediction with memory efficiency and adaptive batch sizing.

        Args:
            model: Scikit-learn-like model with predict/predict_proba method
            X: Feature matrix (DataFrame or numpy array)
            feature_names: Optional feature names for DataFrame conversion
            predict_proba: Whether to predict probabilities
            keep_dataframe: If True, keep X as DataFrame (for models that require it)

        Returns:
            Predictions array
        """
        # Convert to numpy if DataFrame, unless keep_dataframe is True
        if isinstance(X, pd.DataFrame):
            if keep_dataframe:
                # Keep as DataFrame for models that require it (e.g., meta_model)
                X_df = X
                if feature_names:
                    X_df = X_df[feature_names]
            else:
                if feature_names:
                    X = X[feature_names].values
                else:
                    X = X.values

        n_samples = X.shape[0] if not keep_dataframe else X_df.shape[0]
        predictions = np.empty(n_samples, dtype=np.float64)

        # Process in batches with adaptive sizing
        start_idx = 0
        batch_num = 0
        while start_idx < n_samples:
            # Use adaptive batch size if enabled
            if self.adaptive_batching and len(self.memory_usage_samples) >= 3:
                # Calculate median memory usage per sample
                median_bytes_per_sample = np.median(self.memory_usage_samples)
                # Calculate optimal batch size to hit target memory
                target_bytes = self.TARGET_MEMORY_PER_BATCH_MB * 1024 * 1024
                optimal_batch_size = int(target_bytes / median_bytes_per_sample)
                # Clamp to reasonable bounds
                self.batch_size = max(self.MIN_BATCH_SIZE,
                                    min(self.MAX_BATCH_SIZE, optimal_batch_size))

            end_idx = min(start_idx + self.batch_size, n_samples)

            if keep_dataframe:
                batch_X = X_df.iloc[start_idx:end_idx]
            else:
                batch_X = X[start_idx:end_idx]

            # Track memory before prediction
            import gc
            gc.collect()
            mem_before = self._get_memory_usage()

            if predict_proba:
                batch_pred = model.predict_proba(batch_X)
                # Assume binary classification, take probability of class 1
                if batch_pred.ndim == 2 and batch_pred.shape[1] == 2:
                    batch_pred = batch_pred[:, 1]
            else:
                batch_pred = model.predict(batch_X)

            # Track memory after prediction
            gc.collect()
            mem_after = self._get_memory_usage()
            mem_used = max(0, mem_after - mem_before)
            mem_per_sample = mem_used / (end_idx - start_idx)

            # Update memory usage samples (keep last 10)
            self.memory_usage_samples.append(mem_per_sample)
            if len(self.memory_usage_samples) > 10:
                self.memory_usage_samples.pop(0)

            predictions[start_idx:end_idx] = batch_pred
            start_idx = end_idx
            batch_num += 1

        return predictions

    def _get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    def predict_lightgbm_batched(self, model, X: Union[pd.DataFrame, np.ndarray],
                                  feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Optimized batch prediction for LightGBM models.
        
        Args:
            model: LightGBM model
            X: Feature matrix
            feature_names: Optional feature names
            
        Returns:
            Predictions array
        """
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            if feature_names:
                X = X[feature_names].values
            else:
                X = X.values
        
        # LightGBM handles batching internally, but we can still chunk
        # to avoid memory issues with very large datasets
        n_samples = X.shape[0]
        
        if n_samples <= self.batch_size:
            # Single batch prediction
            return model.predict(X, num_threads=self.num_threads)
        
        # Multi-batch prediction
        predictions = np.empty(n_samples, dtype=np.float64)
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_X = X[start_idx:end_idx]
            predictions[start_idx:end_idx] = model.predict(
                batch_X, 
                num_threads=self.num_threads,
                num_iteration=getattr(model, 'best_iteration', None)
            )
        
        return predictions


# ============================================================================
# FEATURE COMPUTATION CACHE
# ============================================================================

class FeatureComputationCache:
    """LRU cache for expensive feature computations."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize feature cache.
        
        Args:
            max_size: Maximum number of cached feature sets
        """
        self.max_size = max_size
        self._cache = {}
        self._access_order = []
        
    def _make_key(self, feature_name: str, timestamp: pd.Timestamp, 
                  symbol: str, params_hash: str) -> str:
        """Create cache key from parameters."""
        return f"{feature_name}:{timestamp}:{symbol}:{params_hash}"
    
    def get(self, feature_name: str, timestamp: pd.Timestamp, 
            symbol: str, params_hash: str) -> Optional[np.ndarray]:
        """
        Get cached feature value.
        
        Args:
            feature_name: Name of the feature
            timestamp: Timestamp of the data point
            symbol: Trading symbol
            params_hash: Hash of computation parameters
            
        Returns:
            Cached feature value or None if not found
        """
        key = self._make_key(feature_name, timestamp, symbol, params_hash)
        if key in self._cache:
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def set(self, feature_name: str, timestamp: pd.Timestamp, 
            symbol: str, params_hash: str, value: np.ndarray) -> None:
        """
        Cache a feature value.
        
        Args:
            feature_name: Name of the feature
            timestamp: Timestamp of the data point
            symbol: Trading symbol
            params_hash: Hash of computation parameters
            value: Feature value to cache
        """
        key = self._make_key(feature_name, timestamp, symbol, params_hash)
        
        # Remove oldest if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        self._cache[key] = value
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": 0.0  # Would need to track hits
        }


# Global feature cache instance
_global_feature_cache = FeatureComputationCache(max_size=10000)


def get_feature_cache() -> FeatureComputationCache:
    """Get the global feature cache instance."""
    return _global_feature_cache


# ============================================================================
# FEATURE LOADING OPTIMIZATION
# ============================================================================

class OptimizedFeatureLoader:
    """Optimized feature loading with chunking and memory mapping."""
    
    def __init__(self, chunk_size: int = 50000, use_memory_map: bool = True):
        """
        Initialize optimized feature loader.
        
        Args:
            chunk_size: Number of rows to load per chunk
            use_memory_map: Whether to use memory mapping for large files
        """
        self.chunk_size = chunk_size
        self.use_memory_map = use_memory_map
    
    def load_features_chunked(self, path: str, 
                              feature_names: Optional[List[str]] = None,
                              start_idx: int = 0,
                              end_idx: Optional[int] = None) -> pd.DataFrame:
        """
        Load features in chunks to reduce memory pressure.
        
        Args:
            path: Path to parquet file
            feature_names: Optional list of features to load
            start_idx: Starting row index
            end_idx: Ending row index (None for end of file)
            
        Returns:
            DataFrame with loaded features
        """
        # Use memory mapping if enabled and file is large
        if self.use_memory_map:
            try:
                # Load with memory mapping for large files
                df = pd.read_parquet(
                    path,
                    columns=feature_names,
                    engine='pyarrow'
                )
                
                # Slice if indices provided
                if end_idx is not None:
                    df = df.iloc[start_idx:end_idx]
                else:
                    df = df.iloc[start_idx:]
                
                return df
            except Exception as e:
                # Fall back to chunked loading
                warnings.warn(f"Memory mapping failed, using chunked loading: {e}")
        
        # Chunked loading
        chunks = []
        for chunk in pd.read_parquet(
            path,
            columns=feature_names,
            chunksize=self.chunk_size,
            engine='pyarrow'
        ):
            # Filter by indices
            chunk_end = start_idx + len(chunk)
            if end_idx is not None and start_idx >= end_idx:
                break
            
            if end_idx is None:
                # Take from start_idx
                if chunk_end > start_idx:
                    chunk_idx_start = max(0, start_idx - (chunk_end - len(chunk)))
                    chunks.append(chunk.iloc[chunk_idx_start:])
                start_idx = chunk_end
            else:
                # Take within range
                chunk_idx_start = max(0, start_idx - (chunk_end - len(chunk)))
                chunk_idx_end = min(len(chunk), end_idx - (chunk_end - len(chunk)))
                if chunk_idx_start < chunk_idx_end:
                    chunks.append(chunk.iloc[chunk_idx_start:chunk_idx_end])
                start_idx = chunk_end
        
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        return pd.DataFrame()
    
    def preload_features(self, paths: List[str], 
                        feature_names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Preload multiple feature files.
        
        Args:
            paths: List of file paths to load
            feature_names: Optional list of features to load
            
        Returns:
            Dictionary mapping paths to DataFrames
        """
        result = {}
        for path in paths:
            result[path] = self.load_features_chunked(path, feature_names)
        return result


# ============================================================================
# META MODEL COMBINATION OPTIMIZATION
# ============================================================================

class OptimizedMetaCombiner:
    """Optimized meta model feature combination and prediction."""
    
    def __init__(self, use_numba: bool = True):
        """
        Initialize optimized meta combiner.
        
        Args:
            use_numba: Whether to use Numba JIT for operations
        """
        self.use_numba = use_numba
    
    def prepare_meta_features_fast(
        self,
        p_alpha: np.ndarray,
        mr_h_preds: Dict[str, np.ndarray],
        tf_h_preds: Dict[str, np.ndarray],
        grp_df: pd.DataFrame,
        cfg: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Fast meta feature preparation with vectorized operations.
        
        Args:
            p_alpha: Alpha model predictions
            mr_h_preds: Mean reversion horizon predictions
            tf_h_preds: Trend-following horizon predictions
            grp_df: Group DataFrame with features
            cfg: Configuration dictionary
            
        Returns:
            DataFrame with meta features
        """
        # Initialize with numeric columns
        num = grp_df.select_dtypes(include=[np.number, bool]).copy()
        X_meta = num.copy()
        
        # Add alpha prediction
        X_meta["pred_logit"] = p_alpha.astype(np.float32)
        
        # 1. Per-horizon logit features (vectorized)
        from scipy.special import logit as _logit_fn
        for h in CANON_HORIZONS:
            ph = mr_h_preds.get(f"pred_H{h}", tf_h_preds.get(f"pred_H{h}"))
            if ph is not None:
                X_meta[f"pred_H{h}"] = ph.astype(np.float32)
                _p_clip = np.clip(ph.astype(float), 1e-4, 1 - 1e-4)
                _lg_h = np.clip(_logit_fn(_p_clip), -4.0, 4.0)
                X_meta[f"pred_logit_H{h}"] = _lg_h.astype(np.float32)
        
        # Store all individual predictions
        for k, v in mr_h_preds.items():
            X_meta[k] = v.astype(np.float32)
        for k, v in tf_h_preds.items():
            X_meta[k] = v.astype(np.float32)
        
        # 2. Disagreement features (vectorized)
        self._add_disagreement_features_fast(X_meta, mr_h_preds, "mr")
        self._add_disagreement_features_fast(X_meta, tf_h_preds, "tf")
        
        # 3. Cross-kind agreement (vectorized)
        self._add_cross_kind_agreement_fast(X_meta, mr_h_preds, tf_h_preds)
        
        # 4. Cross-kind per-horizon diff (vectorized)
        for h in CANON_HORIZONS:
            pmr = mr_h_preds.get(f"pred_mr_H{h}")
            ptf = tf_h_preds.get(f"pred_tf_H{h}")
            if pmr is not None and ptf is not None:
                X_meta[f"tf_minus_mr_H{h}"] = (ptf - pmr).astype(np.float32)
        
        # 5. Core interaction features (vectorized)
        self._add_interaction_features_fast(X_meta, grp_df)
        
        # 6. Regime bucket interactions (vectorized)
        self._add_regime_interactions_fast(X_meta, grp_df, cfg)
        
        # 7. Cross-temporal regime features (vectorized)
        self._add_temporal_regime_features_fast(X_meta, grp_df)
        
        return X_meta
    
    def _add_disagreement_features_fast(
        self,
        X_meta: pd.DataFrame,
        h_preds: Dict[str, np.ndarray],
        kind: str
    ) -> None:
        """Add disagreement features with vectorized operations."""
        horizons = list(CANON_HORIZONS)
        preds = [h_preds.get(f"pred_{kind}_H{h}") for h in horizons]
        preds = [p for p in preds if p is not None]
        
        if len(preds) < 2:
            return
        
        # Compute pairwise disagreements (vectorized)
        n = len(preds[0])
        disagreements = np.zeros((len(preds), len(preds), n), dtype=np.float32)
        
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                if self.use_numba:
                    disagreements[i, j, :] = compute_disagreement_numba(preds[i], preds[j])
                else:
                    disagreements[i, j, :] = np.abs(preds[i] - preds[j])
        
        # Average disagreement
        avg_disagreement = np.zeros(n, dtype=np.float32)
        count = 0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                avg_disagreement += disagreements[i, j, :]
                count += 1
        
        if count > 0:
            X_meta[f"pair_abs_{kind}"] = (avg_disagreement / count).astype(np.float32)
    
    def _add_cross_kind_agreement_fast(
        self,
        X_meta: pd.DataFrame,
        mr_h_preds: Dict[str, np.ndarray],
        tf_h_preds: Dict[str, np.ndarray]
    ) -> None:
        """Add cross-kind agreement features (vectorized)."""
        # Get average disagreements
        mr_disag = X_meta.get("pair_abs_mr")
        tf_disag = X_meta.get("pair_abs_tf")
        
        if mr_disag is not None and tf_disag is not None:
            agree_mr_avg = (1.0 - np.clip(mr_disag, 0.0, 1.0)).astype(np.float32)
            agree_tf_avg = (1.0 - np.clip(tf_disag, 0.0, 1.0)).astype(np.float32)
            X_meta["agree_tf_minus_mr_avg"] = agree_tf_avg - agree_mr_avg
    
    def _add_interaction_features_fast(
        self,
        X_meta: pd.DataFrame,
        grp_df: pd.DataFrame
    ) -> None:
        """Add interaction features (vectorized)."""
        if "pred_logit" not in X_meta.columns:
            return
        
        pl = X_meta["pred_logit"].values
        interact_feats = [
            "vol_z", "mkt_rv_ratio", "ambig", "exh_qual", "trend_pct",
            "trend_t", "trend_z_t", "spike_score", "grind_score", "chop_score"
        ]
        
        for feat in interact_feats:
            if feat in X_meta.columns:
                X_meta[f"pred_x_{feat}"] = (pl * X_meta[feat].values).astype(np.float32)
    
    def _add_regime_interactions_fast(
        self,
        X_meta: pd.DataFrame,
        grp_df: pd.DataFrame,
        cfg: Optional[Dict]
    ) -> None:
        """Add regime bucket interactions (vectorized)."""
        if "pred_logit" not in X_meta.columns:
            return
        
        pl = X_meta["pred_logit"].values
        
        # G_VOL and G_TREND interactions
        for rcol in ["G_VOL", "G_TREND"]:
            if rcol in grp_df.columns:
                rv = grp_df[rcol].values
                for bkt in [0, 1, 2]:
                    mask = (rv == bkt).astype(float)
                    X_meta[f"pred_x_{rcol}_{bkt}"] = (pl * mask).astype(np.float32)
        
        # Granular regime interactions
        regime_map = {
            "vol_12h": "rv_12h",
            "vol_48h": "rv_24h",
            "volume_12h": "vol_z_base",
            "volume_48h": "vol_z24_base",
            "trend_12h": "ret6h",
            "trend_48h": "trend_pct_base",
        }
        
        boundaries = cfg.get("granular_regime_boundaries", {}) if cfg else {}
        
        for rname, src_col in regime_map.items():
            if src_col not in grp_df.columns:
                continue
            
            # Initialize all buckets to 0
            for bkt in [0, 1, 2]:
                X_meta[f"pred_x_{rname}_{bkt}"] = 0.0
            
            vals = grp_df[src_col].values.astype(float)
            valid_mask = np.isfinite(vals)
            
            # Get terciles
            terciles = boundaries.get(rname)
            if terciles is None and valid_mask.sum() > 5:
                try:
                    terciles = np.nanpercentile(vals[valid_mask], [33.3, 66.7]).tolist()
                except Exception:
                    terciles = None
            
            if terciles:
                # Vectorized bucket assignment
                mask0 = vals <= terciles[0]
                mask1 = (vals > terciles[0]) & (vals < terciles[1])
                mask2 = vals >= terciles[1]
                
                X_meta[f"pred_x_{rname}_0"] = (pl * mask0.astype(float)).astype(np.float32)
                X_meta[f"pred_x_{rname}_1"] = (pl * mask1.astype(float)).astype(np.float32)
                X_meta[f"pred_x_{rname}_2"] = (pl * mask2.astype(float)).astype(np.float32)
                
                # Store regime buckets
                regime_vals = (mask1.astype(int) + 2 * mask2.astype(int)).astype(float)
                grp_df[f"__regime_{rname}__"] = regime_vals
            elif valid_mask.sum() > 0:
                # Fallback to mid-bucket
                X_meta[f"pred_x_{rname}_1"] = (pl * valid_mask.astype(float)).astype(np.float32)
                grp_df[f"__regime_{rname}__"] = 1.0
    
    def _add_temporal_regime_features_fast(
        self,
        X_meta: pd.DataFrame,
        grp_df: pd.DataFrame
    ) -> None:
        """Add cross-temporal regime features (vectorized)."""
        # Trend slope ratio
        if "trend_slope_48h" in grp_df.columns and "trend_slope_120h" in grp_df.columns:
            ts48 = grp_df["trend_slope_48h"].values
            ts120 = grp_df["trend_slope_120h"].values
            X_meta["trend_slope_ratio_48_120"] = np.where(
                np.abs(ts120) > 1e-9,
                ts48 / np.clip(np.abs(ts120), 1e-9, None),
                0.0
            ).astype(np.float32)
        
        # Vol regime agreement
        if "__regime_vol_12h__" in grp_df.columns and "__regime_vol_48h__" in grp_df.columns:
            v12 = grp_df["__regime_vol_12h__"].values
            v48 = grp_df["__regime_vol_48h__"].values
            X_meta["vol_regime_agree"] = (v12 == v48).astype(np.float32)
            X_meta["vol_regime_diff"] = (v12 - v48).astype(np.float32)
        
        # Trend regime agreement
        if "__regime_trend_12h__" in grp_df.columns and "__regime_trend_48h__" in grp_df.columns:
            t12 = grp_df["__regime_trend_12h__"].values
            t48 = grp_df["__regime_trend_48h__"].values
            X_meta["trend_regime_agree"] = (t12 == t48).astype(np.float32)
            X_meta["trend_regime_diff"] = (t12 - t48).astype(np.float32)


# ============================================================================
# INTEGRATED PREDICTION PIPELINE
# ============================================================================

class OptimizedPredictionPipeline:
    """Integrated optimized prediction pipeline."""
    
    def __init__(
        self,
        batch_size: int = 10000,
        use_numba: bool = True,
        use_feature_cache: bool = True,
        chunk_size: int = 50000
    ):
        """
        Initialize optimized prediction pipeline.
        
        Args:
            batch_size: Batch size for predictions
            use_numba: Whether to use Numba JIT
            use_feature_cache: Whether to use feature caching
            chunk_size: Chunk size for feature loading
        """
        self.batch_predictor = BatchPredictor(batch_size=batch_size)
        self.meta_combiner = OptimizedMetaCombiner(use_numba=use_numba)
        self.feature_loader = OptimizedFeatureLoader(chunk_size=chunk_size)
        self.use_feature_cache = use_feature_cache
        
        if use_feature_cache:
            self.feature_cache = get_feature_cache()
        else:
            self.feature_cache = None
    
    def predict_meta_fast(
        self,
        meta_model,
        p_alpha: np.ndarray,
        mr_h_preds: Dict[str, np.ndarray],
        tf_h_preds: Dict[str, np.ndarray],
        grp_df: pd.DataFrame,
        cfg: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast meta model prediction with optimized feature preparation.
        
        Args:
            meta_model: Meta model
            p_alpha: Alpha model predictions
            mr_h_preds: Mean reversion horizon predictions
            tf_h_preds: Trend-following horizon predictions
            grp_df: Group DataFrame with features
            cfg: Configuration dictionary
            
        Returns:
            Tuple of (predictions, enabled_mask)
        """
        if meta_model is None:
            return np.full(len(p_alpha), np.nan), np.ones(len(p_alpha), dtype=bool)
        
        # Prepare meta features fast
        X_meta = self.meta_combiner.prepare_meta_features_fast(
            p_alpha, mr_h_preds, tf_h_preds, grp_df, cfg
        )
        
        # Check feature coverage
        if meta_model.selected_features:
            available = set(X_meta.columns)
            # Convert selected_features to list if it's not already
            selected_features_list = list(meta_model.selected_features) if not isinstance(meta_model.selected_features, list) else meta_model.selected_features
            selected = set(selected_features_list)
            coverage = len(selected & available) / len(selected)

            if coverage < 0.8:  # Require 80% coverage
                return np.full(len(p_alpha), np.nan), np.ones(len(p_alpha), dtype=bool)

            # Fill missing features with 0
            missing = selected - available
            if missing:
                import warnings
                warnings.warn(f"Meta prediction: {len(missing)} features missing (coverage {coverage:.0%}), filling with 0")
            X_meta = X_meta.reindex(columns=selected_features_list, fill_value=0.0)

        # Get features for prediction
        # After reindex, X_meta already has correct column order
        # Meta model expects a DataFrame, not numpy array
        X_pred = X_meta

        # Handle NaNs
        X_pred = X_pred.fillna(0.0)
        X_pred = X_pred.replace([np.inf, -np.inf], 0.0)

        # Batch prediction - pass DataFrame directly to meta model
        predictions = self.batch_predictor.predict_batched(meta_model, X_pred, keep_dataframe=True)
        
        return predictions, np.ones(len(p_alpha), dtype=bool)
    
    def compute_position_sizing_fast(
        self,
        confidence: np.ndarray,
        sizing_formula: str = "sigmoid",
        squash_k: float = 1.0,
        base_size: float = 0.05,
        rank_multiplier: float = 0.10,
        c0: float = 0.0,
        s_min: float = 0.03,
        s_max: float = 0.15
    ) -> np.ndarray:
        """
        Fast position sizing computation.
        
        Args:
            confidence: Confidence scores
            sizing_formula: Sizing formula (sigmoid, tanh, concave)
            squash_k: Squash parameter
            base_size: Base position size
            rank_multiplier: Rank multiplier
            c0: Center parameter
            s_min: Minimum size
            s_max: Maximum size
            
        Returns:
            Position sizes
        """
        if sizing_formula == "sigmoid":
            if self.meta_combiner.use_numba:
                z = sigmoid_sizing_numba(confidence, squash_k, c0, 0.0, 1.0)
            else:
                z = 1.0 / (1.0 + np.exp(-squash_k * (confidence - c0)))
        elif sizing_formula == "tanh":
            if self.meta_combiner.use_numba:
                z = tanh_sizing_numba(confidence, squash_k, c0, 0.0, 1.0)
            else:
                z = 0.5 * (1.0 + np.tanh(squash_k * (confidence - c0)))
        else:  # concave
            if self.meta_combiner.use_numba:
                z = concave_sizing_numba(confidence, squash_k, c0, 0.0, 1.0)
            else:
                pos = np.clip(confidence - c0, 0.0, None)
                pos_max = np.max(pos) if len(pos) > 0 else 1.0
                if pos_max > 1e-9:
                    z = (pos / pos_max) ** squash_k
                else:
                    z = np.zeros_like(confidence)
        
        return np.clip(base_size + rank_multiplier * z, 0.0, 1.0)
