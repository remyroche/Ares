"""Feature cache service for storing generated feature matrices.

This module provides a reusable cache utility that stores feature matrices on
 disk so that expensive feature generation steps can be skipped on subsequent
 runs when the same configuration is used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class FeatureCacheService:
    """Persist feature matrices keyed by pipeline configuration."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        *,
        subdirectory: Optional[str] = None,
    ) -> None:
        env_dir = os.getenv("FEATURE_CACHE_DIR")
        root_dir = Path(env_dir) if env_dir else Path("artifacts") / "feature_cache"

        if base_dir is not None:
            root_dir = Path(base_dir)

        if subdirectory:
            root_dir = root_dir / subdirectory

        self.base_dir = root_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__).getChild("FeatureCacheService")

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        return (symbol or "").lower()

    @staticmethod
    def normalize_timeframe(timeframe: str) -> str:
        return (timeframe or "").lower()

    @classmethod
    def build_key(
        cls,
        symbol: str,
        timeframe: str,
        feature_bank_version: str,
        lookback_config_hash: str,
    ) -> str:
        normalized_symbol = cls.normalize_symbol(symbol)
        normalized_timeframe = cls.normalize_timeframe(timeframe)
        version = feature_bank_version or "unknown"
        lookback_hash = lookback_config_hash or "default"
        return "__".join([normalized_symbol, normalized_timeframe, version, lookback_hash])

    @staticmethod
    def compute_config_hash(config: Any) -> str:
        """Create a stable hash for the provided configuration object."""

        if config is None:
            return "default"

        if is_dataclass(config):
            config = asdict(config)
        elif hasattr(config, "to_dict") and callable(config.to_dict):
            try:
                config = config.to_dict()
            except Exception:  # pragma: no cover - fallback path
                config = dict(config) if isinstance(config, dict) else config

        if isinstance(config, dict):
            serializable = json.dumps(config, sort_keys=True, default=str)
        else:
            serializable = json.dumps(config, default=str, sort_keys=True)

        return hashlib.sha256(serializable.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Cache I/O
    # ------------------------------------------------------------------
    def _artifact_path(self, cache_key: str, artifact_type: str) -> Path:
        safe_type = artifact_type.replace("/", "_")
        return self.base_dir / safe_type / f"{cache_key}.parquet"

    def load(self, cache_key: str, artifact_type: str = "features") -> Optional[pd.DataFrame]:
        path = self._artifact_path(cache_key, artifact_type)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            self.logger.debug("Loaded cached %s from %s", artifact_type, path)
            return df
        except Exception as exc:  # pragma: no cover - log and ignore corrupt cache
            self.logger.warning("Failed to load cached %s from %s: %s", artifact_type, path, exc)
            return None

    def save(self, cache_key: str, data: pd.DataFrame, artifact_type: str = "features") -> None:
        if data is None or data.empty:
            self.logger.debug(
                "Skipping cache save for key %s (artifact %s) because data is empty", cache_key, artifact_type
            )
            return

        path = self._artifact_path(cache_key, artifact_type)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data.to_parquet(path)
            self.logger.debug("Saved %s cache artifact to %s", artifact_type, path)
        except Exception as exc:  # pragma: no cover - log but do not raise
            self.logger.warning("Failed to persist cached %s to %s: %s", artifact_type, path, exc)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def clear(self, cache_key: Optional[str] = None, artifact_type: Optional[str] = None) -> None:
        if cache_key is None:
            if self.base_dir.exists():
                for path in self.base_dir.glob("**/*.parquet"):
                    path.unlink(missing_ok=True)
            return

        types = [artifact_type] if artifact_type else [p.name for p in self.base_dir.iterdir() if p.is_dir()]
        for type_name in types:
            path = self._artifact_path(cache_key, type_name)
            path.unlink(missing_ok=True)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
