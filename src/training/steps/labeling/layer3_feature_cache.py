"""
Layer3 Feature Cache for Layer2 Reuse

This module provides caching infrastructure to persist Layer3 meta-features
(including NN embeddings) so they can be reused by Layer2 in subsequent runs.

Key features:
- Cache meta-features to parquet with metadata
- Load cached features with index alignment
- Support for NN embeddings caching
- Versioning and validation
"""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

try:
    from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint_error
except ImportError:
    def tprint_info(*args, **kwargs): print(*args)
    def tprint_warning(*args, **kwargs): print(*args)
    def tprint_success(*args, **kwargs): print(*args)
    def tprint_error(*args, **kwargs): print(*args)


# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

DEFAULT_CACHE_DIR = Path("cache/layer3_features")
CACHE_VERSION = "1.0.0"


def _compute_data_hash(
    market_data: pd.DataFrame,
    n_samples: int = 1000,
) -> str:
    """Compute a hash of market data for cache validation."""
    try:
        # Sample data for hashing (full hash too slow)
        if len(market_data) > n_samples:
            step = len(market_data) // n_samples
            sample = market_data.iloc[::step]
        else:
            sample = market_data
        
        # Hash key columns
        hash_cols = ["close", "high", "low", "volume"]
        hash_cols = [c for c in hash_cols if c in sample.columns]
        
        if not hash_cols:
            return "no_hash_cols"
        
        data_str = sample[hash_cols].to_json()
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    except Exception:
        return "hash_error"


def _get_cache_path(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    cache_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Get cache file paths for features and metadata."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = f"layer3_features_{symbol}_{exchange}_{timeframe}_{direction}"
    features_path = cache_dir / f"{base_name}.parquet"
    metadata_path = cache_dir / f"{base_name}_metadata.json"
    
    return features_path, metadata_path


# =============================================================================
# CACHE SAVE/LOAD
# =============================================================================

def save_layer3_features_to_cache(
    meta_features: pd.DataFrame,
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    market_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[Path] = None,
    nn_embeddings: Optional[pd.DataFrame] = None,
) -> Optional[Path]:
    """
    Save Layer3 meta-features to cache for Layer2 reuse.
    
    Args:
        meta_features: DataFrame with meta-features from Layer3.
        symbol: Trading symbol.
        exchange: Exchange name.
        timeframe: Timeframe string.
        direction: Trade direction.
        market_data: Original market data (for hash validation).
        config: HPO config (for metadata).
        cache_dir: Cache directory path.
        nn_embeddings: Optional separate NN embeddings DataFrame.
        
    Returns:
        Path to saved cache file, or None on failure.
    """
    try:
        features_path, metadata_path = _get_cache_path(
            symbol, exchange, timeframe, direction, cache_dir
        )
        
        # Combine meta_features with nn_embeddings if provided separately
        if nn_embeddings is not None and not nn_embeddings.empty:
            # Align indices
            nn_aligned = nn_embeddings.reindex(meta_features.index)
            # Add columns that don't already exist
            new_cols = [c for c in nn_aligned.columns if c not in meta_features.columns]
            if new_cols:
                meta_features = pd.concat([meta_features, nn_aligned[new_cols]], axis=1)
        
        # Save features
        meta_features.to_parquet(features_path)
        
        # Build metadata
        metadata = {
            "version": CACHE_VERSION,
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "direction": direction,
            "n_rows": int(len(meta_features)),
            "n_cols": int(len(meta_features.columns)),
            "columns": list(meta_features.columns),
            "nn_embed_cols": [c for c in meta_features.columns if c.startswith("nn_embed_")],
            "index_start": str(meta_features.index.min()) if len(meta_features) > 0 else None,
            "index_end": str(meta_features.index.max()) if len(meta_features) > 0 else None,
            "created_at": datetime.utcnow().isoformat(),
            "data_hash": _compute_data_hash(market_data) if market_data is not None else None,
        }
        
        # Add config subset for validation
        if config is not None:
            metadata["config_subset"] = {
                "enable_nn_sequence_embeddings": config.get("meta_feature_engineering", {}).get("enable_nn_sequence_embeddings"),
                "nn_embed_dim": config.get("meta_feature_engineering", {}).get("nn_sequence_encoder", {}).get("embed_dim"),
                "nn_seq_len": config.get("meta_feature_engineering", {}).get("nn_sequence_encoder", {}).get("seq_len"),
            }
        
        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        tprint_success(
            f"[layer3_cache] Saved {len(meta_features)} rows, {len(meta_features.columns)} cols "
            f"(incl. {len(metadata['nn_embed_cols'])} nn_embed_*) to {features_path}"
        )
        
        return features_path
        
    except Exception as e:
        tprint_warning(f"[layer3_cache] Failed to save cache: {e}")
        return None


def load_layer3_features_from_cache(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    target_index: Optional[pd.Index] = None,
    market_data: Optional[pd.DataFrame] = None,
    validate_hash: bool = True,
    cache_dir: Optional[Path] = None,
    max_age_hours: Optional[float] = None,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load Layer3 meta-features from cache.
    
    Args:
        symbol: Trading symbol.
        exchange: Exchange name.
        timeframe: Timeframe string.
        direction: Trade direction.
        target_index: Index to align loaded features to.
        market_data: Market data for hash validation.
        validate_hash: Whether to validate data hash.
        cache_dir: Cache directory path.
        max_age_hours: Maximum cache age in hours (None = no limit).
        
    Returns:
        (features_df, metadata_dict) or (None, {}) on failure.
    """
    try:
        features_path, metadata_path = _get_cache_path(
            symbol, exchange, timeframe, direction, cache_dir
        )
        
        if not features_path.exists() or not metadata_path.exists():
            tprint_info(f"[layer3_cache] No cache found at {features_path}")
            return None, {}
        
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Validate version
        if metadata.get("version") != CACHE_VERSION:
            tprint_warning(
                f"[layer3_cache] Version mismatch: cache={metadata.get('version')}, current={CACHE_VERSION}"
            )
            return None, metadata
        
        # Validate age
        if max_age_hours is not None:
            try:
                created_at = datetime.fromisoformat(metadata.get("created_at", ""))
                age_hours = (datetime.utcnow() - created_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    tprint_warning(
                        f"[layer3_cache] Cache too old: {age_hours:.1f}h > {max_age_hours}h"
                    )
                    return None, metadata
            except Exception:
                pass
        
        # Validate hash
        if validate_hash and market_data is not None:
            current_hash = _compute_data_hash(market_data)
            cached_hash = metadata.get("data_hash")
            if cached_hash and current_hash != cached_hash:
                tprint_warning(
                    f"[layer3_cache] Data hash mismatch: cache={cached_hash}, current={current_hash}"
                )
                return None, metadata
        
        # Load features
        features = pd.read_parquet(features_path)
        
        # Align to target index if provided
        if target_index is not None:
            features = features.reindex(target_index)
            # Fill NaNs for numeric columns
            for col in features.columns:
                if pd.api.types.is_numeric_dtype(features[col]):
                    features[col] = features[col].fillna(0.0)
        
        nn_cols = [c for c in features.columns if c.startswith("nn_embed_")]
        tprint_success(
            f"[layer3_cache] Loaded {len(features)} rows, {len(features.columns)} cols "
            f"(incl. {len(nn_cols)} nn_embed_*) from {features_path}"
        )
        
        return features, metadata
        
    except Exception as e:
        tprint_warning(f"[layer3_cache] Failed to load cache: {e}")
        return None, {}


def get_nn_embeddings_from_cache(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    target_index: Optional[pd.Index] = None,
    cache_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """
    Load only NN embedding columns from cache.
    
    Args:
        symbol: Trading symbol.
        exchange: Exchange name.
        timeframe: Timeframe string.
        direction: Trade direction.
        target_index: Index to align to.
        cache_dir: Cache directory path.
        
    Returns:
        DataFrame with nn_embed_* columns only, or None.
    """
    features, metadata = load_layer3_features_from_cache(
        symbol, exchange, timeframe, direction,
        target_index=target_index,
        validate_hash=False,
        cache_dir=cache_dir,
    )
    
    if features is None:
        return None
    
    nn_cols = [c for c in features.columns if c.startswith("nn_embed_")]
    if not nn_cols:
        tprint_info("[layer3_cache] No nn_embed_* columns in cache")
        return None
    
    return features[nn_cols]


def invalidate_cache(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    cache_dir: Optional[Path] = None,
) -> bool:
    """
    Delete cached features for a given configuration.
    
    Returns:
        True if cache was deleted, False otherwise.
    """
    try:
        features_path, metadata_path = _get_cache_path(
            symbol, exchange, timeframe, direction, cache_dir
        )
        
        deleted = False
        if features_path.exists():
            features_path.unlink()
            deleted = True
        if metadata_path.exists():
            metadata_path.unlink()
            deleted = True
        
        if deleted:
            tprint_info(f"[layer3_cache] Invalidated cache for {symbol}/{exchange}/{timeframe}/{direction}")
        
        return deleted
        
    except Exception as e:
        tprint_warning(f"[layer3_cache] Failed to invalidate cache: {e}")
        return False


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def merge_cached_features_with_new(
    new_features: pd.DataFrame,
    cached_features: Optional[pd.DataFrame],
    prefer_cached_nn: bool = True,
) -> pd.DataFrame:
    """
    Merge newly computed features with cached features.
    
    Args:
        new_features: Newly computed meta-features.
        cached_features: Cached features from previous run.
        prefer_cached_nn: If True, use cached nn_embed_* over newly computed.
        
    Returns:
        Merged DataFrame.
    """
    if cached_features is None or cached_features.empty:
        return new_features
    
    # Align indices
    cached_aligned = cached_features.reindex(new_features.index)
    
    # Identify column categories
    new_nn_cols = [c for c in new_features.columns if c.startswith("nn_embed_")]
    cached_nn_cols = [c for c in cached_aligned.columns if c.startswith("nn_embed_")]
    
    result = new_features.copy()
    
    if prefer_cached_nn and cached_nn_cols:
        # Use cached NN embeddings (they're expensive to compute)
        for col in cached_nn_cols:
            if col in result.columns:
                # Replace with cached values where available
                mask = cached_aligned[col].notna()
                result.loc[mask, col] = cached_aligned.loc[mask, col]
            else:
                # Add cached column
                result[col] = cached_aligned[col]
        
        tprint_info(f"[layer3_cache] Merged {len(cached_nn_cols)} cached nn_embed_* columns")
    
    # Add any other cached columns that don't exist in new features
    other_cached = [c for c in cached_aligned.columns if c not in result.columns and not c.startswith("nn_embed_")]
    if other_cached:
        for col in other_cached:
            result[col] = cached_aligned[col]
        tprint_info(f"[layer3_cache] Added {len(other_cached)} additional cached columns")
    
    return result


def should_use_cached_features(
    config: Dict[str, Any],
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    cache_dir: Optional[Path] = None,
) -> bool:
    """
    Determine if cached features should be used based on config and cache state.
    
    Args:
        config: HPO configuration.
        symbol, exchange, timeframe, direction: Cache key.
        cache_dir: Cache directory.
        
    Returns:
        True if cached features should be loaded and used.
    """
    # Check config flag
    use_cache = bool(config.get("use_layer3_feature_cache", True))
    if not use_cache:
        return False
    
    # Check if cache exists
    features_path, metadata_path = _get_cache_path(
        symbol, exchange, timeframe, direction, cache_dir
    )
    
    if not features_path.exists():
        return False
    
    # Check max age
    max_age = config.get("layer3_cache_max_age_hours")
    if max_age is not None:
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            created_at = datetime.fromisoformat(metadata.get("created_at", ""))
            age_hours = (datetime.utcnow() - created_at).total_seconds() / 3600
            if age_hours > float(max_age):
                return False
        except Exception:
            pass
    
    return True


# =============================================================================
# NN EMBEDDINGS SPECIFIC CACHING
# =============================================================================

def save_nn_embeddings_to_cache(
    nn_embeddings: pd.DataFrame,
    symbol: str,
    exchange: str,
    timeframe: str,
    cache_dir: Optional[Path] = None,
    encoder_config: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Save NN embeddings to a dedicated cache file.
    
    This is separate from the full Layer3 cache for flexibility.
    """
    try:
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR / "nn_embeddings"
        
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Build filename with encoder config hash
        config_str = json.dumps(encoder_config or {}, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        filename = f"nn_embed_{symbol}_{exchange}_{timeframe}_{config_hash}.parquet"
        filepath = cache_dir / filename
        
        nn_embeddings.to_parquet(filepath)
        
        # Save metadata
        metadata = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "encoder_config": encoder_config,
            "n_rows": int(len(nn_embeddings)),
            "n_cols": int(len(nn_embeddings.columns)),
            "columns": list(nn_embeddings.columns),
            "created_at": datetime.utcnow().isoformat(),
        }
        
        meta_path = cache_dir / f"nn_embed_{symbol}_{exchange}_{timeframe}_{config_hash}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        tprint_success(f"[nn_cache] Saved {len(nn_embeddings)} rows to {filepath}")
        return filepath
        
    except Exception as e:
        tprint_warning(f"[nn_cache] Failed to save: {e}")
        return None


def load_nn_embeddings_from_cache(
    symbol: str,
    exchange: str,
    timeframe: str,
    target_index: Optional[pd.Index] = None,
    cache_dir: Optional[Path] = None,
    encoder_config: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Load NN embeddings from dedicated cache.
    """
    try:
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR / "nn_embeddings"
        
        cache_dir = Path(cache_dir)
        
        # Build filename
        config_str = json.dumps(encoder_config or {}, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        filename = f"nn_embed_{symbol}_{exchange}_{timeframe}_{config_hash}.parquet"
        filepath = cache_dir / filename
        
        if not filepath.exists():
            tprint_info(f"[nn_cache] No cache found at {filepath}")
            return None
        
        nn_embeddings = pd.read_parquet(filepath)
        
        # Align to target index
        if target_index is not None:
            nn_embeddings = nn_embeddings.reindex(target_index).fillna(0.0)
        
        tprint_success(f"[nn_cache] Loaded {len(nn_embeddings)} rows from {filepath}")
        return nn_embeddings
        
    except Exception as e:
        tprint_warning(f"[nn_cache] Failed to load: {e}")
        return None
