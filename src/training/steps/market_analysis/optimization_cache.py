"""
Optimization Result Caching and Persistence

This module provides caching and persistence for optimization results
to avoid redundant optimization and enable result reuse.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import pickle
import hashlib
from pathlib import Path
import logging

from src.utils.logger import get_logger
from src.training.steps.pre_training.profit_labeling.consolidated_profit_labeler import MultiHorizonConfig

@dataclass
class CachedOptimizationResult:
    """Cached optimization result."""
    model_type: str
    data_hash: str
    optimization_config: Dict[str, Any]
    optimal_horizons: Dict[str, int]
    optimal_targets: Dict[str, float]
    objective_score: float
    validation_score: float
    performance_metrics: Dict[str, float]
    optimization_time: float
    timestamp: datetime
    expires_at: datetime

    def is_expired(self) -> bool:
        """Check if cached result is expired."""
        return datetime.now() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['expires_at'] = self.expires_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedOptimizationResult':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)

class OptimizationCache:
    """
    Optimization result cache with persistence.
    """

    def __init__(self, cache_dir: str = "optimization_cache",
                 cache_duration_hours: int = 24,
                 max_cache_size: int = 100):
        """Initialize optimization cache."""
        self.cache_dir = Path(cache_dir)
        self.cache_duration_hours = cache_duration_hours
        self.max_cache_size = max_cache_size
        self.logger = get_logger('OptimizationCache')

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing cache
        self.cache = self._load_cache()

        self.logger.info(f'🔧 Optimization cache initialized: {len(self.cache)} cached results')

    def get_cached_result(self,
                         model_type: str,
                         data_hash: str,
                         optimization_config: Dict[str, Any]) -> Optional[CachedOptimizationResult]:
        """
        Get cached optimization result.

        Args:
            model_type: Type of model (analyst/tactician)
            data_hash: Hash of input data
            optimization_config: Optimization configuration

        Returns:
            CachedOptimizationResult if found and not expired, None otherwise
        """
        cache_key = self._generate_cache_key(model_type, data_hash, optimization_config)

        if cache_key in self.cache:
            cached_result = self.cache[cache_key]

            if not cached_result.is_expired():
                self.logger.info(f'📋 Using cached optimization result for {model_type}')
                return cached_result
            else:
                self.logger.info(f'⏰ Cached result expired for {model_type}, removing from cache')
                del self.cache[cache_key]
                self._save_cache()

        return None

    def cache_result(self,
                    model_type: str,
                    data_hash: str,
                    optimization_config: Dict[str, Any],
                    optimal_horizons: Dict[str, int],
                    optimal_targets: Dict[str, float],
                    objective_score: float,
                    validation_score: float,
                    performance_metrics: Dict[str, float],
                    optimization_time: float) -> None:
        """
        Cache optimization result.

        Args:
            model_type: Type of model (analyst/tactician)
            data_hash: Hash of input data
            optimization_config: Optimization configuration
            optimal_horizons: Optimal time horizons
            optimal_targets: Optimal profit targets
            objective_score: Optimization objective score
            validation_score: Validation score
            performance_metrics: Performance metrics
            optimization_time: Time taken for optimization
        """
        cache_key = self._generate_cache_key(model_type, data_hash, optimization_config)

        # Create cached result
        cached_result = CachedOptimizationResult(
            model_type=model_type,
            data_hash=data_hash,
            optimization_config=optimization_config,
            optimal_horizons=optimal_horizons,
            optimal_targets=optimal_targets,
            objective_score=objective_score,
            validation_score=validation_score,
            performance_metrics=performance_metrics,
            optimization_time=optimization_time,
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=self.cache_duration_hours)
        )

        # Add to cache
        self.cache[cache_key] = cached_result

        # Clean up cache if too large
        if len(self.cache) > self.max_cache_size:
            self._cleanup_cache()

        # Save cache
        self._save_cache()

        self.logger.info(f'💾 Cached optimization result for {model_type}')

    def clear_cache(self, model_type: Optional[str] = None) -> None:
        """
        Clear cache.

        Args:
            model_type: Optional model type to clear, clears all if None
        """
        if model_type is None:
            self.cache.clear()
            self.logger.info('🗑️ Cleared all cached results')
        else:
            keys_to_remove = [key for key, result in self.cache.items()
                            if result.model_type == model_type]
            for key in keys_to_remove:
                del self.cache[key]
            self.logger.info(f'🗑️ Cleared cached results for {model_type}')

        self._save_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_results = len(self.cache)
        expired_results = sum(1 for result in self.cache.values() if result.is_expired())
        valid_results = total_results - expired_results

        # Group by model type
        model_stats = {}
        for result in self.cache.values():
            if not result.is_expired():
                model_type = result.model_type
                if model_type not in model_stats:
                    model_stats[model_type] = 0
                model_stats[model_type] += 1

        return {
            'total_results': total_results,
            'valid_results': valid_results,
            'expired_results': expired_results,
            'model_stats': model_stats,
            'cache_size_mb': self._get_cache_size_mb()
        }

    def _generate_cache_key(self, model_type: str, data_hash: str,
                           optimization_config: Dict[str, Any]) -> str:
        """Generate cache key."""
        config_str = json.dumps(optimization_config, sort_keys=True)
        key_string = f"{model_type}_{data_hash}_{config_str}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _load_cache(self) -> Dict[str, CachedOptimizationResult]:
        """Load cache from disk."""
        cache_file = self.cache_dir / "optimization_cache.json"

        if not cache_file.exists():
            return {}

        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            cache = {}
            for key, data in cache_data.items():
                try:
                    cache[key] = CachedOptimizationResult.from_dict(data)
                except Exception as e:
                    self.logger.warning(f'⚠️ Error loading cached result {key}: {e}')
                    continue

            return cache

        except Exception as e:
            self.logger.warning(f'⚠️ Error loading cache: {e}')
            return {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        cache_file = self.cache_dir / "optimization_cache.json"

        try:
            cache_data = {}
            for key, result in self.cache.items():
                if not result.is_expired():
                    cache_data[key] = result.to_dict()

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

        except Exception as e:
            self.logger.error(f'❌ Error saving cache: {e}')

    def _cleanup_cache(self) -> None:
        """Clean up cache by removing oldest entries."""
        # Sort by timestamp and remove oldest entries
        sorted_results = sorted(self.cache.items(), key=lambda x: x[1].timestamp)

        # Remove oldest 20% of entries
        remove_count = len(sorted_results) // 5
        for key, _ in sorted_results[:remove_count]:
            del self.cache[key]

        self.logger.info(f'🧹 Cleaned up {remove_count} old cache entries')

    def _get_cache_size_mb(self) -> float:
        """Get cache size in MB."""
        try:
            cache_file = self.cache_dir / "optimization_cache.json"
            if cache_file.exists():
                return cache_file.stat().st_size / (1024 * 1024)
            return 0.0
        except Exception:
            return 0.0

def calculate_data_hash(data: pd.DataFrame) -> str:
    """Calculate hash of data for caching."""
    try:
        # Use data shape, columns, and sample of data for hash
        data_info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'sample': data.head(100).to_dict() if len(data) > 0 else {}
        }

        data_str = json.dumps(data_info, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    except Exception:
        # Fallback to simple hash
        return hashlib.md5(str(data.shape).encode()).hexdigest()
