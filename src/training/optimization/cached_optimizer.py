# src/training/optimization/cached_optimizer.py

"""Cached Optimizer for efficient parameter optimization with caching and warm start."""

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

    def __init__(self, config: dict[str, Any]) -> None:
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
        except Exception:
            self.print(warning("Could not load cache metadata: {e}"))
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
        except Exception:
            self.print(error("Could not save cache metadata: {e}"))
            return False

    def _generate_cache_key(self, optimization_config: dict[str, Any]) -> str:
        """Generate cache key based on optimization configuration."""
        config_str = json.dumps(optimization_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="cached results retrieval",
    )
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

        except Exception:
            self.print(warning("Error validating cache: {e}"))
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

        except Exception:
            self.print(warning("Error getting warm start parameters: {e}"))
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

        except Exception:
            self.print(warning("Error calculating config similarity: {e}"))
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

        except Exception:
            self.print(error("Error caching optimization results: {e}"))
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

        except Exception:
            self.print(error("Error running optimization with warm start: {e}"))
            return None

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="cache cleanup",
    )