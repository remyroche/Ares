# src/training/optimization/cached_optimizer.py

"""Cached Optimizer for efficient parameter optimization with caching and warm start."""

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import optuna

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    warning,
)


@dataclass
class CacheConfig:
    """Configuration for caching optimization results."""

    cache_dir: str = "cache/optimization"
    cache_ttl_hours: int = 24
    max_cache_size_mb: int = 100
    enable_warm_start: bool = True
    warm_start_threshold: float = 0.8  # Similarity threshold for warm start


class CachedOptimizer:
    """Implements caching for optimization efficiency with warm start capabilities."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize cached optimizer."""
        self.config = config
        self.logger = system_logger.getChild("CachedOptimizer")
        self.cache_config = CacheConfig(**config.get("cache_config", {}))

        # Ensure cache directory exists
        os.makedirs(self.cache_config.cache_dir, exist_ok=True)

        # Cache storage
        self.cache_metadata_file = os.path.join(
            self.cache_config.cache_dir, "metadata.json",
        )
        self.cache_metadata = self._load_cache_metadata()

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="cache metadata loading",
    )
    def _load_cache_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from file."""
        try:
            if os.path.exists(self.cache_metadata_file):
                with open(self.cache_metadata_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.warning(warning(f"Could not load cache metadata: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="cache metadata saving",
    )
    def _save_cache_metadata(self) -> bool:
        """Save cache metadata to file."""
        try:
            with open(self.cache_metadata_file, "w") as f:
                json.dump(self.cache_metadata, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(error(f"Could not save cache metadata: {e}"))
            return False

    def _generate_cache_key(self, optimization_config: Dict[str, Any]) -> str:
        """Generate cache key based on optimization configuration."""
        config_str = json.dumps(optimization_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> str:
        """Get cache file path for given key."""
        return os.path.join(self.cache_config.cache_dir, f"{cache_key}.pkl")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="cached results retrieval",
    )
    def get_cached_optimization_results(
        self,
        optimization_config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Get cached optimization results if available and valid."""
        try:
            cache_key = self._generate_cache_key(optimization_config)
            cache_file = self._get_cache_file_path(cache_key)

            # Check if cache exists and is valid
            if not os.path.exists(cache_file):
                return None

            # Check cache age
            cache_age = datetime.now() - datetime.fromtimestamp(
                os.path.getmtime(cache_file),
            )
            if cache_age > timedelta(hours=self.cache_config.cache_ttl_hours):
                self.logger.info(f"Cache expired for key {cache_key}")
                return None

            # Load cached results
            with open(cache_file, "rb") as f:
                cached_results = pickle.load(f)

            self.logger.info(f"Retrieved cached results for key {cache_key}")
            return cached_results

        except Exception as e:
            self.logger.warning(warning(f"Error retrieving cached results: {e}"))
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="cache validation",
    )
    def is_cache_valid(self, cached_results: Dict[str, Any]) -> bool:
        """Check if cached results are valid."""
        try:
            # Check if results have required fields
            required_fields = ["best_params", "best_value", "optimization_history"]
            if not all(field in cached_results for field in required_fields):
                return False

            # Check if results are recent enough
            if "timestamp" in cached_results:
                result_age = datetime.now() - datetime.fromisoformat(
                    cached_results["timestamp"],
                )
                if result_age > timedelta(hours=self.cache_config.cache_ttl_hours):
                    return False

            return True

        except Exception as e:
            self.logger.warning(warning(f"Error validating cache: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="warm start parameters retrieval",
    )
    def get_warm_start_parameters(
        self, optimization_config: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Get warm start parameters from cached results."""
        try:
            if not self.cache_config.enable_warm_start:
                return None

            # Get cached results
            cached_results = self.get_cached_optimization_results(optimization_config)
            if not cached_results or not self.is_cache_valid(cached_results):
                return None

            # Calculate similarity with current config
            similarity = self._calculate_config_similarity(
                cached_results.get("optimization_config", {}),
                optimization_config,
            )

            if similarity >= self.cache_config.warm_start_threshold:
                self.logger.info(
                    f"Using warm start parameters (similarity: {similarity:.2f})",
                )
                return cached_results.get("best_params", {})

            return None

        except Exception as e:
            self.logger.warning(warning(f"Error getting warm start parameters: {e}"))
            return None

    def _calculate_config_similarity(
        self, config1: Dict[str, Any],
        config2: Dict[str, Any],
    ) -> float:
        """Calculate similarity between two optimization configurations."""
        try:
            # Convert configs to comparable format
            config1_str = json.dumps(config1, sort_keys=True)
            config2_str = json.dumps(config2, sort_keys=True)

            # Simple string similarity (can be enhanced with more sophisticated methods)
            if config1_str == config2_str:
                return 1.0

            # Calculate similarity based on common keys
            common_keys = set(config1.keys()) & set(config2.keys())
            if not common_keys:
                return 0.0

            similar_values = 0
            for key in common_keys:
                if config1[key] == config2[key]:
                    similar_values += 1

            return similar_values / len(common_keys)

        except Exception as e:
            self.logger.warning(warning(f"Error calculating config similarity: {e}"))
            return 0.0

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="cache storage",
    )
    def cache_optimization_results(
        self,
        optimization_config: Dict[str, Any],
        results: Dict[str, Any],
    ) -> bool:
        """Cache optimization results."""
        try:
            cache_key = self._generate_cache_key(optimization_config)
            cache_file = self._get_cache_file_path(cache_key)

            # Add metadata to results
            results["timestamp"] = datetime.now().isoformat()
            results["optimization_config"] = optimization_config
            results["cache_key"] = cache_key

            # Save results to cache
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)

            # Update metadata
            self.cache_metadata[cache_key] = {
                "timestamp": results["timestamp"],
                "file_size": os.path.getsize(cache_file),
                "config_hash": cache_key,
            }

            # Save metadata
            self._save_cache_metadata()

            self.logger.info(f"Cached optimization results for key {cache_key}")
            return True

        except Exception as e:
            self.logger.error(error(f"Error caching optimization results: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="optimization with warm start",
    )
    def run_optimization_with_warm_start(
        self,
        optimization_config: Dict[str, Any],
        objective_function,
        n_trials: int = 100,
    ) -> Dict[str, Any]:
        """Run optimization with warm start capabilities."""
        try:
            # Check for cached results first
            cached_results = self.get_cached_optimization_results(optimization_config)
            if cached_results and self.is_cache_valid(cached_results):
                self.logger.info("Using cached optimization results")
                return cached_results

            # Get warm start parameters
            warm_start_params = self.get_warm_start_parameters(optimization_config)

            # Create study
            study_name = f"cached_optimization_{int(datetime.now().timestamp())}"
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                storage=None,
            )

            # Add warm start if available
            if warm_start_params:
                study.enqueue_trial(warm_start_params)
                self.logger.info("Added warm start trial")

            # Run optimization
            study.optimize(objective_function, n_trials=n_trials)

            # Prepare results
            results = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "optimization_history": [
                    {
                        "trial_number": trial.number,
                        "value": trial.value,
                        "params": trial.params,
                    }
                    for trial in study.trials
                ],
                "n_trials": len(study.trials),
            }

            # Cache results
            self.cache_optimization_results(optimization_config, results)

            return results

        except Exception as e:
            self.logger.error(error(f"Error in optimization with warm start: {e}"))
            return {}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="cache cleanup",
    )
    def cleanup_expired_cache(self) -> Optional[Dict[str, Any]]:
        """Clean up expired cache files."""
        try:
            current_time = datetime.now()
            cleaned_files = 0
            total_size_freed = 0

            # Check all cache files
            for filename in os.listdir(self.cache_config.cache_dir):
                if filename.endswith(".pkl"):
                    file_path = os.path.join(self.cache_config.cache_dir, filename)
                    file_age = current_time - datetime.fromtimestamp(os.path.getmtime(file_path))

                    # Remove expired files
                    if file_age > timedelta(hours=self.cache_config.cache_ttl_hours):
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        cleaned_files += 1
                        total_size_freed += file_size

                        # Remove from metadata
                        cache_key = filename.replace(".pkl", "")
                        if cache_key in self.cache_metadata:
                            del self.cache_metadata[cache_key]

            # Save updated metadata
            if cleaned_files > 0:
                self._save_cache_metadata()

            self.logger.info(
                f"Cleaned up {cleaned_files} expired cache files, freed {total_size_freed / 1024 / 1024:.2f} MB",
            )

            return {
                "cleaned_files": cleaned_files,
                "size_freed_mb": total_size_freed / 1024 / 1024,
            }

        except Exception as e:
            self.logger.error(error(f"Error cleaning up cache: {e}"))
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="cache statistics",
    )
    def get_cache_statistics(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        try:
            total_files = len(self.cache_metadata)
            total_size_mb = 0

            # Calculate total size
            for cache_info in self.cache_metadata.values():
                total_size_mb += cache_info.get("file_size", 0) / 1024 / 1024

            # Get cache age distribution
            cache_ages = []
            current_time = datetime.now()
            for cache_info in self.cache_metadata.values():
                if "timestamp" in cache_info:
                    cache_time = datetime.fromisoformat(cache_info["timestamp"])
                    age_hours = (current_time - cache_time).total_seconds() / 3600
                    cache_ages.append(age_hours)

            return {
                "total_files": total_files,
                "total_size_mb": total_size_mb,
                "avg_cache_age_hours": sum(cache_ages) / len(cache_ages) if cache_ages else 0,
                "oldest_cache_hours": max(cache_ages) if cache_ages else 0,
                "newest_cache_hours": min(cache_ages) if cache_ages else 0,
                "cache_dir": self.cache_config.cache_dir,
                "cache_ttl_hours": self.cache_config.cache_ttl_hours,
            }

        except Exception as e:
            self.logger.error(error(f"Error getting cache statistics: {e}"))
            return None


def create_cached_optimizer(config: Optional[Dict[str, Any]] = None) -> CachedOptimizer:
    """Create a cached optimizer instance.

    Args:
        config: Optional configuration dictionary

    Returns:
        CachedOptimizer instance

    """
    if config is None:
        config = {}

    return CachedOptimizer(config)
