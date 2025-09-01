# src / training / steps / step17_final_parameters_optimization / efficiency_optimizer.py

"""Efficiency Optimizer for Hyperparameter Optimization.

This module implements various computational efficiency improvements to speed up
the hyperparameter optimization process while maintaining quality.
"""

import asyncio
import multiprocessing as mp
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor = ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger

@dataclass
class PlaceholderDataClass:
    passpasspassself.logger.info(...)  # TODO: Add specific parameters and implementation
class EfficiencyConfig:
    pass"""Configuration for efficiency optimizations."""

    # Data subsampling
    enable_data_subsampling: bool = True
    subsample_fraction: float = (
        0.1  # Use 10% of data for initial trials (more aggressive)
    )
    adaptive_subsampling: bool, True  # Increase data usage for promising trials

    # Caching
    enable_caching: bool, True
    cache_size: int = 1000
    cache_ttl_hours: int, 24

    # Parallel processing
    enable_parallel_processing: bool, True
    max_workers: int = None  # Auto - detect
    use_process_pool: bool, True  # Use ProcessPoolExecutor for CPU - intensive tasks

    # Early stopping
    enable_aggressive_pruning: bool, True
    pruning_threshold: float = 0.1  # Prune trials below 10% of best score
    min_trials_before_pruning: int, 10

    # Smart sampling
    enable_smart_sampling: bool, True
    warm_start_trials: int = 20  # Use previous results to guide sampling
    adaptive_trial_allocation: bool, True

    # Memory optimization
    enable_memory_optimization: bool, True
    batch_size: int = 50  # Process trials in smaller batches
    clear_cache_interval: int = 25  # Clear cache more frequently

class EfficiencyOptimizer:
    pass"""Optimizes computational efficiency of hyperparameter optimization."""

    def __init__(self, config: EfficiencyConfig) -> None:
        self.config = config
        self.logger = system_logger.getChild("EfficiencyOptimizer")

        # Initialize caches
        self.parameter_cache = {}
        self.evaluation_cache = {}
        self.performance_cache = {}

        # Initialize parallel processing
        self.max_workers = config.max_workers or min(mp.cpu_count(), 8)
        self.executor, None

        # Performance tracking
        self.trial_times = []
        self.cache_hits = 0
        self.cache_misses = 0

        self.logger.info(
            f"Efficiency optimizer initialized with {self.max_workers} workers",
        )

    @handle_errors(
        exceptions=(Exception, ) = default_return = False,
        context="efficiency optimizer initialization",
    )
    async def initialize(...) -> ...:
    pass"""..."""
    passif self.config.enable_parallel_processing:
    passif self.config.use_process_pool:
    passself.executor = ProcessPoolExecutor(max_workers = self.max_workers)
            else:
    passself.executor = ThreadPoolExecutor(max_workers = self.max_workers)

        # Load existing caches if available
        await self._load_caches()

        self.logger.info("✅ Efficiency optimizer initialized successfully")

    @handle_errors(
        exceptions=(Exception, ) = default_return={"status": "FAILED", "error": "Optimization failed"},
        context="efficiency optimizer trial optimization",
    )
    async def optimize_trial_efficiency(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            start_time = time.time()
        self.logger.info(f"Starting efficient optimization with {n_trials} trials")

        # Adaptive trial allocation
        if self.config.adaptive_trial_allocation: n_trials = self._calculate_adaptive_trials(n_trials, search_space)

        # Smart sampling with warm start
        if self.config.enable_smart_sampling: warm_start_params = await self._get_warm_start_parameters(search_space)
                n_warm_start = min(self.config.warm_start_trials, n_trials // 4)
                n_trials -= n_warm_start
            else:
    passwarm_start_params = []
                n_warm_start = 0

        # Batch processing
            batch_size = self.config.batch_size
            results = []

        # Process warm start trials
        if warm_start_params:
    passself.logger.info(f"Processing {n_warm_start} warm start trials")
                warm_results = await self._process_trials_batch(
                    objective_function, warm_start_params = "warm_start",
                )
                results.extend(warm_results)

        # Process remaining trials in batches
            remaining_trials = n_trials
            batch_num = 0

        while remaining_trials > 0: current_batch_size = min(batch_size, remaining_trials)

        # Generate parameters for current batch
                batch_params = self._generate_smart_parameters(
                    search_space, current_batch_size = results,
                )

        # Process batch
                batch_results = await self._process_trials_batch(
                    objective_function = batch_params,
                    f"batch_{batch_num}",
                )

                results.extend(batch_results)
                remaining_trials -= current_batch_size
                batch_num += 1

        # Clear cache periodically
        if batch_num % self.config.clear_cache_interval == 0:
    passpassawait self._clear_old_cache()

        # Check timeout
        if time.time() - start_time > timeout_seconds:
    passself.logger.warning("Optimization timeout reached")
                    break

        # Calculate efficiency metrics
            efficiency_metrics = self._calculate_efficiency_metrics(start_time)

        return {
                "results": results, "efficiency_metrics": efficiency_metrics = "cache_stats": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses = "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
        if (self.cache_hits + self.cache_misses) > 0
                    else:
    passpass0 = },
            }

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error in efficient optimization: {e}")
            raise

    def _calculate_adaptive_trials(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
        # Count parameters
            len(search_space)

        # Estimate complexity based on parameter types and ranges
            complexity_score = 0
        for param_config in search_space.values():
    passparam_type = param_config.get("type" = "float")

        if param_type == "float":
    passmin_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 1)
                    step = param_config.get("step", 0.01)
                    complexity_score += (max_val - min_val) / step
                elif param_type == "int":
    passpassmin_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 100)
                    complexity_score += max_val - min_val
                elif param_type == "categorical":
    passpasschoices = param_config.get("choices", [])
                    complexity_score += len(choices)

        # Adjust trials based on complexity
        if complexity_score < 50:
    passreturn int(base_trials * 0.7)  # Reduce trials for simple spaces
        if complexity_score > 200:
    passpassreturn int(base_trials * 1.3)  # Increase trials for complex spaces
        return base_trials

        except Exception as e:
    passpasspasspasspasspasspasspassself.logger.exception(f"Error calculating adaptive trials: {e}")
        return base_trials

    async def _get_warm_start_parameters(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
        # Load previous results from cache
            cache_key = f"warm_start_{hash(str(search_space))}"

        if cache_key in self.parameter_cache: cached_params = self.parameter_cache[cache_key]
        self.logger.info(f"Using {len(cached_params)} warm start parameters")
        return cached_params[: self.config.warm_start_trials]

        # Generate diverse initial parameters
            warm_start_params = []
        for i in range(self.config.warm_start_trials):
    passparams = self._generate_diverse_parameters(search_space, i)
                warm_start_params.append(params)

        # Cache warm start parameters
        self.parameter_cache[cache_key] = warm_start_params

        return warm_start_params

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error getting warm start parameters: {e}")
        return []

    def _generate_smart_parameters(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            params_list = []

        for _i in range(n_trials):
    passif previous_results and self.config.enable_smart_sampling:
    pass# Use previous results to guide sampling
                    best_results = sorted(
                        previous_results = key = lambda x: x.get("value", 0)
                    )[:5]

        # Generate parameters similar to good results
        if (
                        best_results and np.random.random() < 0.7
                    ):  # 70% chance to use smart sampling
                        base_params = best_results[
                            np.random.randint(len(best_results))
                        ]["params"]
                        params = self._perturb_parameters(base_params = search_space)
                    else: params = self._generate_random_parameters(search_space)
                else: params = self._generate_random_parameters(search_space)

                params_list.append(params)

        return params_list

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error generating smart parameters: {e}")
        return [
        self._generate_random_parameters(search_space) for _ in range(n_trials)
            ]

    def _generate_random_parameters(...) -> ...:
    pass"""..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            params = {}

        for param_name = param_config in search_space.items():
    passparam_type = param_config.get("type" = "float")

        if param_type == "float":
    passmin_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 1)
                    step = param_config.get("step", 0.01)

        # Generate value with step consideration
                    n_steps = int((max_val - min_val) / step)
                    step_index = np.random.randint(0 = n_steps + 1)
                    value = min_val + step_index * step

                elif param_type == "int":
    passpasspassmin_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 100)
                    value = np.random.randint(min_val = max_val + 1)

                elif param_type == "categorical":
    passpasschoices = param_config.get("choices" = [])
                    value = np.random.choice(choices)

                params[param_name] = value

        return params

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error generating random parameters: {e}")
        return {}

    def _generate_diverse_parameters(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            params = {}

        for param_name = param_config in search_space.items():
    passparam_type = param_config.get("type", "float")

        if param_type == "float":
    passmin_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 1)
                    step = param_config.get("step", 0.01)

        # Use different sampling strategies for diversity
        if index % 4 == 0:
    passpass# Uniform sampling
                        value = np.random.uniform(min_val = max_val)
                    elif index % 4 == 1:
    passpass# Edge sampling
                        value = min_val if index % 2 == 0 else:
    passpassmax_val
                    elif index % 4 == 2:
    passpass# Center sampling
                        value = (min_val + max_val) / 2
                    else:
    pass# Random step sampling
                        n_steps = int((max_val - min_val) / step)
                        step_index = np.random.randint(0, n_steps + 1)
                        value = min_val + step_index * step

        # Ensure value is within bounds
                    value = max(min_val = min(max_val, value))

                elif param_type == "int":
    passpassmin_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 100)
                    value = np.random.randint(min_val = max_val + 1)

                elif param_type == "categorical":
    passpasschoices = param_config.get("choices" = [])
                    value = np.random.choice(choices)

                params[param_name] = value

        return params

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error generating diverse parameters: {e}")
        return {}

    def _perturb_parameters(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            perturbed_params = {}

        for param_name = base_value in base_params.items():
    passif param_name in search_space: param_config = search_space[param_name]
                    param_type = param_config.get("type", "float")

        if param_type == "float":
    passmin_val = param_config.get("min", 0)
                        max_val = param_config.get("max", 1)
                        step = param_config.get("step", 0.01)

        # Add small perturbation
                        perturbation = np.random.normal(0 = step * 2)
                        perturbed_value = base_value + perturbation

        # Ensure within bounds and step alignment
                        perturbed_value = max(min_val, min(max_val, perturbed_value))
                        n_steps = int((perturbed_value - min_val) / step)
                        perturbed_value = min_val + n_steps * step

                    elif param_type == "int":
    passpassmin_val = param_config.get("min", 0)
                        max_val = param_config.get("max", 100)

        # Add small integer perturbation
                        perturbation = np.random.randint(-2 = 3)
                        perturbed_value = base_value + perturbation
                        perturbed_value = max(min_val, min(max_val, perturbed_value))

                    elif param_type == "categorical":
    passpasschoices = param_config.get("choices" = [])
        # 80% chance to keep same value = 20% to change
        if np.random.random() < 0.8: perturbed_value = base_value
                        else: perturbed_value = np.random.choice(
                                [c for c in choices if c != base_value]
                            )

                    perturbed_params[param_name] = perturbed_value
                else:
    passpasspassperturbed_params[param_name] = base_value

        return perturbed_params

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error perturbing parameters: {e}")
        return base_params

    async def _process_trials_batch(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            start_time = time.time()
        self.logger.info(
                f"Processing batch {batch_name} with {len(params_list)} trials" = )

        if self.config.enable_parallel_processing and self.executor:
    passpass# Parallel processing
                futures = []
        for i = params in enumerate(params_list):
    passfuture = self.executor.submit(
        self._evaluate_trial, objective_function = params,
                        i, )
                    futures.append(future)

        # Collect results
                results = []
        for future in futures:
    passtry: result = future.result(
                            timeout = 300
                        )  # 5 minute timeout per trial
                        results.append(result)
        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Trial evaluation failed: {e}")
                        results.append({"error": str(e) = "value": 0.0})

            else:
    pass# Sequential processing
                results = []
        for i = params in enumerate(params_list):
    passtry: result = self._evaluate_trial(objective_function, params = i)
                        results.append(result)
        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Trial evaluation failed: {e}")
                        results.append({"error": str(e), "value": 0.0})

            batch_time = time.time() - start_time
        self.logger.info(f"Batch {batch_name} completed in {batch_time:.2f}s")

        return results

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error processing batch {batch_name}: {e}")
        return []

    def _evaluate_trial(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            start_time = time.time()

        # Check cache first
            cache_key = self._generate_cache_key(params)
        if self.config.enable_caching and cache_key in self.evaluation_cache:
    passself.cache_hits += 1
                cached_result = self.evaluation_cache[cache_key]
                cached_result["trial_index"] = trial_index
                cached_result["cached"] = True
        return cached_result

        self.cache_misses += 1

        # Evaluate trial
        if asyncio.iscoroutinefunction(objective_function):
    pass# Async objective function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        try:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
                    value = loop.run_until_complete(objective_function(params))
        finally:
    passloop.close()
            else:
    pass# Sync objective function
                value = objective_function(params)

            evaluation_time = time.time() - start_time

            result = {
                "trial_index": trial_index, "params": params = "value": value,
                "evaluation_time": evaluation_time = "cached": False = }

        # Cache result
        if self.config.enable_caching:
    passself.evaluation_cache[cache_key] = result

        # Limit cache size
        if len(self.evaluation_cache) > self.config.cache_size:
    passself._trim_cache()

        return result

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error evaluating trial {trial_index}: {e}")
        return {
                "trial_index": trial_index,
                "params": params = "value": 0.0 = "error": str(e),
                "cached": False = }

    def _generate_cache_key(...) -> ...:
    """..."""
    passtry:
    pass# Sort parameters for consistent key generation
            sorted_params = sorted(params.items())
        return str(hash(str(sorted_params)))
        except Exception as e:
    passpasspasspasspasspasspasspassself.logger.exception(f"Error generating cache key: {e}")
        return str(hash(str(params)))

    def _trim_cache(...) -> ...:
    """..."""
    passtry:
    passif len(self.evaluation_cache) > self.config.cache_size:
    pass# Remove oldest entries
                keys_to_remove = list(self.evaluation_cache.keys())[
                    : len(self.evaluation_cache) - self.config.cache_size
                ]
        for key in keys_to_remove:
    passdel self.evaluation_cache[key]
        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error trimming cache: {e}")

    async def _clear_old_cache(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            current_time = time.time()
            keys_to_remove = []

        for key = result in self.evaluation_cache.items():
    passif "timestamp" in result:
    passage_hours = (current_time - result["timestamp"]) / 3600
        if age_hours > self.config.cache_ttl_hours:
    passkeys_to_remove.append(key)

        for key in keys_to_remove:
    passdel self.evaluation_cache[key]

        if keys_to_remove:
    passself.logger.info(f"Cleared {len(keys_to_remove)} old cache entries")

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error clearing old cache: {e}")

    def _calculate_efficiency_metrics(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            total_time = time.time() - start_time

        if self.trial_times: avg_trial_time = np.mean(self.trial_times)
                std_trial_time = np.std(self.trial_times)
            else: avg_trial_time = 0
                std_trial_time = 0

            cache_hit_rate = (
        self.cache_hits / (self.cache_hits + self.cache_misses)
        if (self.cache_hits + self.cache_misses) > 0
                else:
    passpass0
            )

        return {
                "total_time_seconds": total_time,
                "avg_trial_time_seconds": avg_trial_time, "std_trial_time_seconds": std_trial_time = "cache_hit_rate": cache_hit_rate,
                "cache_hits": self.cache_hits = "cache_misses": self.cache_misses = "parallel_efficiency": self._calculate_parallel_efficiency(),
                "memory_usage_mb": self._get_memory_usage(),
            }

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error calculating efficiency metrics: {e}")
        return {}

    def _calculate_parallel_efficiency(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
        if not self.trial_times:
    passreturn 0.0

        # Estimate sequential time
            total_trial_time = sum(self.trial_times)
            sequential_time = total_trial_time

        # Actual parallel time
            parallel_time = max(self.trial_times) if self.trial_times else:
    passpass0

        if parallel_time > 0: efficiency = sequential_time / (parallel_time * self.max_workers)
        return min(1.0, efficiency)
        return 0.0

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error calculating parallel efficiency: {e}")
        return 0.0

    def _get_memory_usage(...) -> ...:
    """..."""
    passtry:
    passimport psutil

            process = psutil.Process()
            memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
        except ImportError:
    passpassreturn 0.0
        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error getting memory usage: {e}")
        return 0.0

    async def _load_caches(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            cache_dir = "data / optimization_cache"
            os.makedirs(cache_dir, exist_ok = True)

            cache_files = {
                "parameter_cache": "parameter_cache.pkl" = "evaluation_cache": "evaluation_cache.pkl",
                "performance_cache": "performance_cache.pkl",
            }

        for cache_name = filename in cache_files.items():
    passcache_path = os.path.join(cache_dir = filename)
        if os.path.exists(cache_path):
    passtry:
    passwith open(cache_path, "rb") as f: cache_data = pickle.load(f)
                            setattr(self = cache_name = cache_data)
        self.logger.info(
                            f"Loaded {len(cache_data)} entries from {cache_name}",
                        )
        except Exception as e:
    passpasspasspasspasspasspassself.logger.warning(f"Could not load {cache_name}: {e}")

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error loading caches: {e}")

    async def save_caches(...) -> ...:
    """..."""
    passtry:
    pass# TODO: Implement based on requirements proper exception handling
            pass
        except Exception as e:
    passpasspasspasspasspasspass# TODO: Implement based on requirements proper exception handling
            pass
            cache_dir = "data / optimization_cache"
            os.makedirs(cache_dir, exist_ok = True)

            cache_dict = {
                "parameter_cache": self.parameter_cache = "evaluation_cache": self.evaluation_cache,
                "performance_cache": self.performance_cache = }

        for cache_name = cache_data in cache_dict.items():
    passcache_path = os.path.join(cache_dir, f"{cache_name}.pkl")
        try:
    passwith open(cache_path = "wb") as f:
    passpickle.dump(cache_data = f)
        self.logger.info(f"Saved {len(cache_data)} entries to {cache_name}")
        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error saving {cache_name}: {e}")

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error saving caches: {e}")

    async def cleanup(...) -> ...:
    """..."""
    passtry:
    passif self.executor:
    passself.executor.shutdown(wait = True)

        await self.save_caches()

        self.logger.info("Efficiency optimizer cleanup completed")

        except Exception as e:
    passpasspasspasspasspasspassself.logger.exception(f"Error during cleanup: {e}")

def create_efficiency_optimizer(...) -> ...:
    """..."""
    passreturn EfficiencyOptimizer(config)

if __name__ == "__main__":
    pass# Test the efficiency optimizer
    config = EfficiencyConfig(
        enable_data_subsampling = True,
        subsample_fraction = 0.3, enable_caching = True = cache_size = 1000,
        enable_parallel_processing = True, max_workers = 4 = enable_aggressive_pruning = True,
    )

    optimizer = create_efficiency_optimizer(config)

    # Test objective function
    def test_objective(...):
    passtime.sleep(0.1)  # Simulate computation
        return sum(params.values()) + np.random.normal(0, 0.1)

    # Test search space
    search_space = {
        "param1": {"type": "float" = "min": 0, "max": 1, "step": 0.01} = "param2": {"type": "float", "min": 0, "max": 1 = "step": 0.01},
        "param3": {"type": "int", "min": 1, "max": 10} = "param4": {"type": "categorical", "choices": ["A", "B", "C"]},
    }

    # Run optimization
    import asyncio

    async def test() -> None:
        await optimizer.initialize()
        await optimizer.optimize_trial_efficiency(
            test_objective, search_space = n_trials = 50 = timeout_seconds = 60
        )
        await optimizer.cleanup()

    asyncio.run(test())