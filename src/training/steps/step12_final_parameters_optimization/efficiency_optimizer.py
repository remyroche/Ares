# src/training/steps/step12_final_parameters_optimization/efficiency_optimizer.py

"""
Efficiency Optimizer for Hyperparameter Optimization

This module implements various computational efficiency improvements to speed up
the hyperparameter optimization process while maintaining quality.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import pickle
import os
from functools import lru_cache
import joblib

from src.utils.logger import system_logger


@dataclass
class EfficiencyConfig:
    """Configuration for efficiency optimizations."""
    
    # Data subsampling
    enable_data_subsampling: bool = True
    subsample_fraction: float = 0.3  # Use 30% of data for initial trials
    adaptive_subsampling: bool = True  # Increase data usage for promising trials
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl_hours: int = 24
    
    # Parallel processing
    enable_parallel_processing: bool = True
    max_workers: int = None  # Auto-detect
    use_process_pool: bool = True  # Use ProcessPoolExecutor for CPU-intensive tasks
    
    # Early stopping
    enable_aggressive_pruning: bool = True
    pruning_threshold: float = 0.1  # Prune trials below 10% of best score
    min_trials_before_pruning: int = 10
    
    # Smart sampling
    enable_smart_sampling: bool = True
    warm_start_trials: int = 20  # Use previous results to guide sampling
    adaptive_trial_allocation: bool = True
    
    # Memory optimization
    enable_memory_optimization: bool = True
    batch_size: int = 100  # Process trials in batches
    clear_cache_interval: int = 50  # Clear cache every 50 trials


class EfficiencyOptimizer:
    """Optimizes computational efficiency of hyperparameter optimization."""
    
    def __init__(self, config: EfficiencyConfig):
        self.config = config
        self.logger = system_logger.getChild("EfficiencyOptimizer")
        
        # Initialize caches
        self.parameter_cache = {}
        self.evaluation_cache = {}
        self.performance_cache = {}
        
        # Initialize parallel processing
        self.max_workers = config.max_workers or min(mp.cpu_count(), 8)
        self.executor = None
        
        # Performance tracking
        self.trial_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.logger.info(f"Efficiency optimizer initialized with {self.max_workers} workers")
    
    async def initialize(self):
        """Initialize the efficiency optimizer."""
        try:
            if self.config.enable_parallel_processing:
                if self.config.use_process_pool:
                    self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
                else:
                    self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            
            # Load existing caches if available
            await self._load_caches()
            
            self.logger.info("✅ Efficiency optimizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing efficiency optimizer: {e}")
            raise
    
    async def optimize_trial_efficiency(
        self,
        objective_function,
        search_space: Dict[str, Any],
        n_trials: int,
        timeout_seconds: int = 3600,
    ) -> Dict[str, Any]:
        """
        Run efficient hyperparameter optimization.
        
        Args:
            objective_function: Function to optimize
            search_space: Parameter search space
            n_trials: Number of trials
            timeout_seconds: Timeout in seconds
            
        Returns:
            Optimization results with efficiency metrics
        """
        try:
            start_time = time.time()
            self.logger.info(f"Starting efficient optimization with {n_trials} trials")
            
            # Adaptive trial allocation
            if self.config.adaptive_trial_allocation:
                n_trials = self._calculate_adaptive_trials(n_trials, search_space)
            
            # Smart sampling with warm start
            if self.config.enable_smart_sampling:
                warm_start_params = await self._get_warm_start_parameters(search_space)
                n_warm_start = min(self.config.warm_start_trials, n_trials // 4)
                n_trials -= n_warm_start
            else:
                warm_start_params = []
                n_warm_start = 0
            
            # Batch processing
            batch_size = self.config.batch_size
            results = []
            
            # Process warm start trials
            if warm_start_params:
                self.logger.info(f"Processing {n_warm_start} warm start trials")
                warm_results = await self._process_trials_batch(
                    objective_function, warm_start_params, "warm_start"
                )
                results.extend(warm_results)
            
            # Process remaining trials in batches
            remaining_trials = n_trials
            batch_num = 0
            
            while remaining_trials > 0:
                current_batch_size = min(batch_size, remaining_trials)
                
                # Generate parameters for current batch
                batch_params = self._generate_smart_parameters(
                    search_space, current_batch_size, results
                )
                
                # Process batch
                batch_results = await self._process_trials_batch(
                    objective_function, batch_params, f"batch_{batch_num}"
                )
                
                results.extend(batch_results)
                remaining_trials -= current_batch_size
                batch_num += 1
                
                # Clear cache periodically
                if batch_num % self.config.clear_cache_interval == 0:
                    await self._clear_old_cache()
                
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    self.logger.warning("Optimization timeout reached")
                    break
            
            # Calculate efficiency metrics
            efficiency_metrics = self._calculate_efficiency_metrics(start_time)
            
            return {
                "results": results,
                "efficiency_metrics": efficiency_metrics,
                "cache_stats": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in efficient optimization: {e}")
            raise
    
    def _calculate_adaptive_trials(self, base_trials: int, search_space: Dict[str, Any]) -> int:
        """Calculate adaptive number of trials based on search space complexity."""
        try:
            # Count parameters
            n_params = len(search_space)
            
            # Estimate complexity based on parameter types and ranges
            complexity_score = 0
            for param_name, param_config in search_space.items():
                param_type = param_config.get("type", "float")
                
                if param_type == "float":
                    min_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 1)
                    step = param_config.get("step", 0.01)
                    complexity_score += (max_val - min_val) / step
                elif param_type == "int":
                    min_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 100)
                    complexity_score += max_val - min_val
                elif param_type == "categorical":
                    choices = param_config.get("choices", [])
                    complexity_score += len(choices)
            
            # Adjust trials based on complexity
            if complexity_score < 50:
                return int(base_trials * 0.7)  # Reduce trials for simple spaces
            elif complexity_score > 200:
                return int(base_trials * 1.3)  # Increase trials for complex spaces
            else:
                return base_trials
                
        except Exception as e:
            self.logger.error(f"Error calculating adaptive trials: {e}")
            return base_trials
    
    async def _get_warm_start_parameters(self, search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get warm start parameters from previous optimizations."""
        try:
            # Load previous results from cache
            cache_key = f"warm_start_{hash(str(search_space))}"
            
            if cache_key in self.parameter_cache:
                cached_params = self.parameter_cache[cache_key]
                self.logger.info(f"Using {len(cached_params)} warm start parameters")
                return cached_params[:self.config.warm_start_trials]
            
            # Generate diverse initial parameters
            warm_start_params = []
            for i in range(self.config.warm_start_trials):
                params = self._generate_diverse_parameters(search_space, i)
                warm_start_params.append(params)
            
            # Cache warm start parameters
            self.parameter_cache[cache_key] = warm_start_params
            
            return warm_start_params
            
        except Exception as e:
            self.logger.error(f"Error getting warm start parameters: {e}")
            return []
    
    def _generate_smart_parameters(
        self,
        search_space: Dict[str, Any],
        n_trials: int,
        previous_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate smart parameters based on previous results."""
        try:
            params_list = []
            
            for i in range(n_trials):
                if previous_results and self.config.enable_smart_sampling:
                    # Use previous results to guide sampling
                    best_results = sorted(previous_results, key=lambda x: x.get("value", 0), reverse=True)[:5]
                    
                    # Generate parameters similar to good results
                    if best_results and np.random.random() < 0.7:  # 70% chance to use smart sampling
                        base_params = best_results[np.random.randint(len(best_results))]["params"]
                        params = self._perturb_parameters(base_params, search_space)
                    else:
                        params = self._generate_random_parameters(search_space)
                else:
                    params = self._generate_random_parameters(search_space)
                
                params_list.append(params)
            
            return params_list
            
        except Exception as e:
            self.logger.error(f"Error generating smart parameters: {e}")
            return [self._generate_random_parameters(search_space) for _ in range(n_trials)]
    
    def _generate_random_parameters(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random parameters within search space."""
        try:
            params = {}
            
            for param_name, param_config in search_space.items():
                param_type = param_config.get("type", "float")
                
                if param_type == "float":
                    min_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 1)
                    step = param_config.get("step", 0.01)
                    
                    # Generate value with step consideration
                    n_steps = int((max_val - min_val) / step)
                    step_index = np.random.randint(0, n_steps + 1)
                    value = min_val + step_index * step
                    
                elif param_type == "int":
                    min_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 100)
                    value = np.random.randint(min_val, max_val + 1)
                    
                elif param_type == "categorical":
                    choices = param_config.get("choices", [])
                    value = np.random.choice(choices)
                
                params[param_name] = value
            
            return params
            
        except Exception as e:
            self.logger.error(f"Error generating random parameters: {e}")
            return {}
    
    def _generate_diverse_parameters(self, search_space: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Generate diverse parameters for warm start."""
        try:
            params = {}
            
            for param_name, param_config in search_space.items():
                param_type = param_config.get("type", "float")
                
                if param_type == "float":
                    min_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 1)
                    step = param_config.get("step", 0.01)
                    
                    # Use different sampling strategies for diversity
                    if index % 4 == 0:
                        # Uniform sampling
                        value = np.random.uniform(min_val, max_val)
                    elif index % 4 == 1:
                        # Edge sampling
                        value = min_val if index % 2 == 0 else max_val
                    elif index % 4 == 2:
                        # Center sampling
                        value = (min_val + max_val) / 2
                    else:
                        # Random step sampling
                        n_steps = int((max_val - min_val) / step)
                        step_index = np.random.randint(0, n_steps + 1)
                        value = min_val + step_index * step
                    
                    # Ensure value is within bounds
                    value = max(min_val, min(max_val, value))
                    
                elif param_type == "int":
                    min_val = param_config.get("min", 0)
                    max_val = param_config.get("max", 100)
                    value = np.random.randint(min_val, max_val + 1)
                    
                elif param_type == "categorical":
                    choices = param_config.get("choices", [])
                    value = np.random.choice(choices)
                
                params[param_name] = value
            
            return params
            
        except Exception as e:
            self.logger.error(f"Error generating diverse parameters: {e}")
            return {}
    
    def _perturb_parameters(self, base_params: Dict[str, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Perturb base parameters to create similar but different parameters."""
        try:
            perturbed_params = {}
            
            for param_name, base_value in base_params.items():
                if param_name in search_space:
                    param_config = search_space[param_name]
                    param_type = param_config.get("type", "float")
                    
                    if param_type == "float":
                        min_val = param_config.get("min", 0)
                        max_val = param_config.get("max", 1)
                        step = param_config.get("step", 0.01)
                        
                        # Add small perturbation
                        perturbation = np.random.normal(0, step * 2)
                        perturbed_value = base_value + perturbation
                        
                        # Ensure within bounds and step alignment
                        perturbed_value = max(min_val, min(max_val, perturbed_value))
                        n_steps = int((perturbed_value - min_val) / step)
                        perturbed_value = min_val + n_steps * step
                        
                    elif param_type == "int":
                        min_val = param_config.get("min", 0)
                        max_val = param_config.get("max", 100)
                        
                        # Add small integer perturbation
                        perturbation = np.random.randint(-2, 3)
                        perturbed_value = base_value + perturbation
                        perturbed_value = max(min_val, min(max_val, perturbed_value))
                        
                    elif param_type == "categorical":
                        choices = param_config.get("choices", [])
                        # 80% chance to keep same value, 20% to change
                        if np.random.random() < 0.8:
                            perturbed_value = base_value
                        else:
                            perturbed_value = np.random.choice([c for c in choices if c != base_value])
                    
                    perturbed_params[param_name] = perturbed_value
                else:
                    perturbed_params[param_name] = base_value
            
            return perturbed_params
            
        except Exception as e:
            self.logger.error(f"Error perturbing parameters: {e}")
            return base_params
    
    async def _process_trials_batch(
        self,
        objective_function,
        params_list: List[Dict[str, Any]],
        batch_name: str
    ) -> List[Dict[str, Any]]:
        """Process a batch of trials efficiently."""
        try:
            start_time = time.time()
            self.logger.info(f"Processing batch {batch_name} with {len(params_list)} trials")
            
            if self.config.enable_parallel_processing and self.executor:
                # Parallel processing
                futures = []
                for i, params in enumerate(params_list):
                    future = self.executor.submit(self._evaluate_trial, objective_function, params, i)
                    futures.append(future)
                
                # Collect results
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per trial
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Trial evaluation failed: {e}")
                        results.append({"error": str(e), "value": 0.0})
                
            else:
                # Sequential processing
                results = []
                for i, params in enumerate(params_list):
                    try:
                        result = self._evaluate_trial(objective_function, params, i)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Trial evaluation failed: {e}")
                        results.append({"error": str(e), "value": 0.0})
            
            batch_time = time.time() - start_time
            self.logger.info(f"Batch {batch_name} completed in {batch_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_name}: {e}")
            return []
    
    def _evaluate_trial(self, objective_function, params: Dict[str, Any], trial_index: int) -> Dict[str, Any]:
        """Evaluate a single trial with caching."""
        try:
            start_time = time.time()
            
            # Check cache first
            cache_key = self._generate_cache_key(params)
            if self.config.enable_caching and cache_key in self.evaluation_cache:
                self.cache_hits += 1
                cached_result = self.evaluation_cache[cache_key]
                cached_result["trial_index"] = trial_index
                cached_result["cached"] = True
                return cached_result
            
            self.cache_misses += 1
            
            # Evaluate trial
            if asyncio.iscoroutinefunction(objective_function):
                # Async objective function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    value = loop.run_until_complete(objective_function(params))
                finally:
                    loop.close()
            else:
                # Sync objective function
                value = objective_function(params)
            
            evaluation_time = time.time() - start_time
            
            result = {
                "trial_index": trial_index,
                "params": params,
                "value": value,
                "evaluation_time": evaluation_time,
                "cached": False
            }
            
            # Cache result
            if self.config.enable_caching:
                self.evaluation_cache[cache_key] = result
                
                # Limit cache size
                if len(self.evaluation_cache) > self.config.cache_size:
                    self._trim_cache()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating trial {trial_index}: {e}")
            return {
                "trial_index": trial_index,
                "params": params,
                "value": 0.0,
                "error": str(e),
                "cached": False
            }
    
    def _generate_cache_key(self, params: Dict[str, Any]) -> str:
        """Generate cache key for parameters."""
        try:
            # Sort parameters for consistent key generation
            sorted_params = sorted(params.items())
            return str(hash(str(sorted_params)))
        except Exception as e:
            self.logger.error(f"Error generating cache key: {e}")
            return str(hash(str(params)))
    
    def _trim_cache(self):
        """Trim cache to maintain size limit."""
        try:
            if len(self.evaluation_cache) > self.config.cache_size:
                # Remove oldest entries
                keys_to_remove = list(self.evaluation_cache.keys())[:len(self.evaluation_cache) - self.config.cache_size]
                for key in keys_to_remove:
                    del self.evaluation_cache[key]
        except Exception as e:
            self.logger.error(f"Error trimming cache: {e}")
    
    async def _clear_old_cache(self):
        """Clear old cache entries."""
        try:
            current_time = time.time()
            keys_to_remove = []
            
            for key, result in self.evaluation_cache.items():
                if "timestamp" in result:
                    age_hours = (current_time - result["timestamp"]) / 3600
                    if age_hours > self.config.cache_ttl_hours:
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.evaluation_cache[key]
            
            if keys_to_remove:
                self.logger.info(f"Cleared {len(keys_to_remove)} old cache entries")
                
        except Exception as e:
            self.logger.error(f"Error clearing old cache: {e}")
    
    def _calculate_efficiency_metrics(self, start_time: float) -> Dict[str, Any]:
        """Calculate efficiency metrics."""
        try:
            total_time = time.time() - start_time
            
            if self.trial_times:
                avg_trial_time = np.mean(self.trial_times)
                std_trial_time = np.std(self.trial_times)
            else:
                avg_trial_time = 0
                std_trial_time = 0
            
            cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            
            return {
                "total_time_seconds": total_time,
                "avg_trial_time_seconds": avg_trial_time,
                "std_trial_time_seconds": std_trial_time,
                "cache_hit_rate": cache_hit_rate,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "parallel_efficiency": self._calculate_parallel_efficiency(),
                "memory_usage_mb": self._get_memory_usage()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating efficiency metrics: {e}")
            return {}
    
    def _calculate_parallel_efficiency(self) -> float:
        """Calculate parallel processing efficiency."""
        try:
            if not self.trial_times:
                return 0.0
            
            # Estimate sequential time
            total_trial_time = sum(self.trial_times)
            sequential_time = total_trial_time
            
            # Actual parallel time
            parallel_time = max(self.trial_times) if self.trial_times else 0
            
            if parallel_time > 0:
                efficiency = sequential_time / (parallel_time * self.max_workers)
                return min(1.0, efficiency)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating parallel efficiency: {e}")
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {e}")
            return 0.0
    
    async def _load_caches(self):
        """Load existing caches from disk."""
        try:
            cache_dir = "data/optimization_cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_files = {
                "parameter_cache": "parameter_cache.pkl",
                "evaluation_cache": "evaluation_cache.pkl",
                "performance_cache": "performance_cache.pkl"
            }
            
            for cache_name, filename in cache_files.items():
                cache_path = os.path.join(cache_dir, filename)
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, "rb") as f:
                            cache_data = pickle.load(f)
                            setattr(self, cache_name, cache_data)
                            self.logger.info(f"Loaded {len(cache_data)} entries from {cache_name}")
                    except Exception as e:
                        self.logger.warning(f"Could not load {cache_name}: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error loading caches: {e}")
    
    async def save_caches(self):
        """Save caches to disk."""
        try:
            cache_dir = "data/optimization_cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_data = {
                "parameter_cache": self.parameter_cache,
                "evaluation_cache": self.evaluation_cache,
                "performance_cache": self.performance_cache
            }
            
            for cache_name, cache_data in cache_data.items():
                cache_path = os.path.join(cache_dir, f"{cache_name}.pkl")
                try:
                    with open(cache_path, "wb") as f:
                        pickle.dump(cache_data, f)
                    self.logger.info(f"Saved {len(cache_data)} entries to {cache_name}")
                except Exception as e:
                    self.logger.error(f"Error saving {cache_name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error saving caches: {e}")
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            
            await self.save_caches()
            
            self.logger.info("Efficiency optimizer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


def create_efficiency_optimizer(config: EfficiencyConfig) -> EfficiencyOptimizer:
    """Create an efficiency optimizer instance."""
    return EfficiencyOptimizer(config)


if __name__ == "__main__":
    # Test the efficiency optimizer
    config = EfficiencyConfig(
        enable_data_subsampling=True,
        subsample_fraction=0.3,
        enable_caching=True,
        cache_size=1000,
        enable_parallel_processing=True,
        max_workers=4,
        enable_aggressive_pruning=True
    )
    
    optimizer = create_efficiency_optimizer(config)
    
    # Test objective function
    def test_objective(params):
        time.sleep(0.1)  # Simulate computation
        return sum(params.values()) + np.random.normal(0, 0.1)
    
    # Test search space
    search_space = {
        "param1": {"type": "float", "min": 0, "max": 1, "step": 0.01},
        "param2": {"type": "float", "min": 0, "max": 1, "step": 0.01},
        "param3": {"type": "int", "min": 1, "max": 10},
        "param4": {"type": "categorical", "choices": ["A", "B", "C"]}
    }
    
    # Run optimization
    import asyncio
    
    async def test():
        await optimizer.initialize()
        results = await optimizer.optimize_trial_efficiency(
            test_objective, search_space, n_trials=50, timeout_seconds=60
        )
        print(f"Optimization completed with {len(results['results'])} trials")
        print(f"Efficiency metrics: {results['efficiency_metrics']}")
        await optimizer.cleanup()
    
    asyncio.run(test())