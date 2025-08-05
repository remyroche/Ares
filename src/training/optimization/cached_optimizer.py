# src/training/optimization/cached_optimizer.py

"""
Cached Optimizer for efficient parameter optimization with caching and warm start.
"""

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import optuna

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


@dataclass
class CacheConfig:
    """Configuration for caching optimization results."""

    cache_dir: str = "cache/optimization"
    cache_ttl_hours: int = 24
    max_cache_size_mb: int = 100
    enable_warm_start: bool = True
    warm_start_threshold: float = 0.8  # Similarity threshold for warm start


class CachedOptimizer:
    """
    Implements caching for optimization efficiency with warm start capabilities.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize cached optimizer."""
        self.config = config
        self.logger = system_logger.getChild("CachedOptimizer")
        self.cache_config = CacheConfig(**config.get("cache_config", {}))

        # Ensure cache directory exists
        os.makedirs(self.cache_config.cache_dir, exist_ok=True)

        # Cache storage
        self.cache_metadata_file = os.path.join(
            self.cache_config.cache_dir,
            "metadata.json",
        )
        self.cache_metadata = self._load_cache_metadata()

    @handle_errors(
        exceptions=(Exception,),
        default_return={},
        context="cache metadata loading",
    )
    def _load_cache_metadata(self) -> dict[str, Any]:
        """Load cache metadata from file."""
        try:
            if os.path.exists(self.cache_metadata_file):
                with open(self.cache_metadata_file) as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.warning(f"Could not load cache metadata: {e}")
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
            self.logger.error(f"Could not save cache metadata: {e}")
            return False

    def _generate_cache_key(self, optimization_config: dict[str, Any]) -> str:
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
        optimization_config: dict[str, Any],
    ) -> dict[str, Any] | None:
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
            self.logger.warning(f"Error retrieving cached results: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="cache validation",
    )
    def is_cache_valid(self, cached_results: dict[str, Any]) -> bool:
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
            self.logger.warning(f"Error validating cache: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="warm start parameters retrieval",
    )
    def get_warm_start_parameters(
        self,
        optimization_config: dict[str, Any],
    ) -> dict[str, Any] | None:
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
            self.logger.warning(f"Error getting warm start parameters: {e}")
            return None

    def _calculate_config_similarity(
        self,
        config1: dict[str, Any],
        config2: dict[str, Any],
    ) -> float:
        """Calculate similarity between two optimization configurations."""
        try:
            # Convert configs to comparable format
            config1_str = json.dumps(config1, sort_keys=True)
            config2_str = json.dumps(config2, sort_keys=True)

            # Simple string similarity (can be enhanced with more sophisticated methods)
            if config1_str == config2_str:
                return 1.0

            # Calculate Jaccard similarity for key sets
            keys1 = set(config1.keys())
            keys2 = set(config2.keys())

            if not keys1 and not keys2:
                return 1.0

            intersection = len(keys1.intersection(keys2))
            union = len(keys1.union(keys2))

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"Error calculating config similarity: {e}")
            return 0.0

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="optimization results caching",
    )
    def cache_optimization_results(
        self,
        optimization_config: dict[str, Any],
        results: dict[str, Any],
    ) -> bool:
        """Cache optimization results."""
        try:
            cache_key = self._generate_cache_key(optimization_config)
            cache_file = self._get_cache_file_path(cache_key)

            # Prepare results for caching
            cache_data = {
                "optimization_config": optimization_config,
                "best_params": results.get("best_params", {}),
                "best_value": results.get("best_value", 0.0),
                "optimization_history": results.get("optimization_history", []),
                "timestamp": datetime.now().isoformat(),
                "cache_key": cache_key,
            }

            # Save to cache file
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            # Update metadata
            self.cache_metadata[cache_key] = {
                "file_path": cache_file,
                "timestamp": datetime.now().isoformat(),
                "config_hash": cache_key,
            }
            self._save_cache_metadata()

            self.logger.info(f"Cached optimization results for key {cache_key}")
            return True

        except Exception as e:
            self.logger.error(f"Error caching optimization results: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="optimization with warm start",
    )
    async def run_optimization_with_warm_start(
        self,
        optimization_config: dict[str, Any],
        objective_function: callable,
    ) -> dict[str, Any] | None:
        """Run optimization with warm start capabilities."""
        try:
            # Check for cached results first
            cached_results = self.get_cached_optimization_results(optimization_config)
            if cached_results and self.is_cache_valid(cached_results):
                self.logger.info("Using cached optimization results")
                return cached_results

            # Get warm start parameters
            warm_start_params = self.get_warm_start_parameters(optimization_config)

            # Create Optuna study with warm start
            study_name = f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if warm_start_params:
                # Create study with warm start
                study = optuna.create_study(
                    study_name=study_name,
                    direction="maximize",
                    storage=None,
                )

                # Add warm start trial
                study.enqueue_trial(warm_start_params)
                self.logger.info(
                    f"Added warm start trial with {len(warm_start_params)} parameters",
                )
            else:
                # Create study without warm start
                study = optuna.create_study(
                    study_name=study_name,
                    direction="maximize",
                    storage=None,
                )

            # Run optimization
            n_trials = optimization_config.get("n_trials", 100)
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
                "study_name": study_name,
                "n_trials": len(study.trials),
            }

            # Cache results
            self.cache_optimization_results(optimization_config, results)

            return results

        except Exception as e:
            self.logger.error(f"Error running optimization with warm start: {e}")
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="cache cleanup",
    )
    async def cleanup_cache(self) -> bool:
        """Clean up expired cache files."""
        try:
            current_time = datetime.now()
            cleaned_files = 0

            for cache_key, metadata in self.cache_metadata.items():
                file_path = metadata.get("file_path")
                if file_path and os.path.exists(file_path):
                    file_age = current_time - datetime.fromtimestamp(
                        os.path.getmtime(file_path),
                    )

                    if file_age > timedelta(hours=self.cache_config.cache_ttl_hours):
                        os.remove(file_path)
                        del self.cache_metadata[cache_key]
                        cleaned_files += 1

            # Save updated metadata
            self._save_cache_metadata()

            self.logger.info(f"Cleaned up {cleaned_files} expired cache files")
            return True

        except Exception as e:
            self.logger.error(f"Error cleaning up cache: {e}")
            return False

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            total_files = len(self.cache_metadata)
            total_size_mb = 0

            for metadata in self.cache_metadata.values():
                file_path = metadata.get("file_path")
                if file_path and os.path.exists(file_path):
                    total_size_mb += os.path.getsize(file_path) / (1024 * 1024)

            return {
                "total_cached_results": total_files,
                "total_cache_size_mb": round(total_size_mb, 2),
                "cache_dir": self.cache_config.cache_dir,
                "cache_ttl_hours": self.cache_config.cache_ttl_hours,
                "enable_warm_start": self.cache_config.enable_warm_start,
            }

        except Exception as e:
            self.logger.error(f"Error getting cache statistics: {e}")
            return {}
