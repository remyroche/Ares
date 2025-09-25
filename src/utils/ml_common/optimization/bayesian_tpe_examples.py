"""
Examples and Tests for Bayesian TPE Optimizer

This module provides comprehensive examples and tests demonstrating how to use
the Bayesian TPE optimizer with automatic grid search integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import time
import logging

from .bayesian_tpe_optimizer import (
    BayesianTPEOptimizer, 
    BayesianTPEConfig, 
    optimize_with_bayesian_tpe,
    create_search_space_from_bounds
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_simple_function_optimization():
    """Example 1: Simple mathematical function optimization."""
    logger.info("🎯 Example 1: Simple function optimization")
    
    def objective_function(params: Dict[str, Any], **kwargs) -> float:
        """Simple objective function to maximize."""
        x = params['x']
        y = params['y']
        # Maximize: -(x-2)^2 - (y-3)^2 (peak at x=2, y=3)
        return -(x - 2)**2 - (y - 3)**2
    
    # Define search space
    search_space = {
        'x': {'type': 'float', 'low': -5.0, 'high': 5.0},
        'y': {'type': 'float', 'low': -5.0, 'high': 5.0}
    }
    
    # Configure optimizer
    config = BayesianTPEConfig(
        n_trials=30,
        coarse_grid_points=3,
        fine_grid_points=5,
        enable_progress_logging=True
    )
    
    # Optimize
    optimizer = BayesianTPEOptimizer(config)
    result = optimizer.optimize(objective_function, search_space)
    
    logger.info(f"✅ Best parameters: {result.best_params}")
    logger.info(f"📊 Best score: {result.best_score:.4f}")
    logger.info(f"⏱️ Optimization time: {result.optimization_time:.2f}s")
    
    return result


def example_2_machine_learning_optimization():
    """Example 2: Machine learning hyperparameter optimization."""
    logger.info("🎯 Example 2: ML hyperparameter optimization")
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)
    
    def objective_function(params: Dict[str, Any], X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """ML objective function using cross-validation."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        
        # Create model with parameters
        model = RandomForestRegressor(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_samples_split=int(params['min_samples_split']),
            random_state=42
        )
        
        # Cross-validation score
        scores = cross_val_score(model, X, y, cv=3, scoring='r2')
        return np.mean(scores)
    
    # Define search space
    search_space = {
        'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
        'max_depth': {'type': 'int', 'low': 3, 'high': 20},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 10}
    }
    
    # Configure optimizer
    config = BayesianTPEConfig(
        n_trials=20,
        coarse_grid_points=4,
        fine_grid_points=6,
        backend='optuna'
    )
    
    # Optimize
    optimizer = BayesianTPEOptimizer(config)
    result = optimizer.optimize(objective_function, search_space, X=X, y=y)
    
    logger.info(f"✅ Best parameters: {result.best_params}")
    logger.info(f"📊 Best score: {result.best_score:.4f}")
    
    return result


def example_3_advanced_configuration():
    """Example 3: Advanced configuration with custom settings."""
    logger.info("🎯 Example 3: Advanced configuration")
    
    def objective_function(params: Dict[str, Any], **kwargs) -> float:
        """Complex objective function with multiple parameters."""
        x1, x2, x3 = params['x1'], params['x2'], params['x3']
        method = params['method']
        
        # Different objective based on method
        if method == 'linear':
            return x1 + x2 + x3
        elif method == 'quadratic':
            return -(x1**2 + x2**2 + x3**2)
        else:  # exponential
            return np.exp(-(x1**2 + x2**2 + x3**2))
    
    # Define search space with mixed types
    search_space = {
        'x1': {'type': 'float', 'low': -2.0, 'high': 2.0},
        'x2': {'type': 'float', 'low': -2.0, 'high': 2.0},
        'x3': {'type': 'int', 'low': 1, 'high': 10},
        'method': {'type': 'categorical', 'choices': ['linear', 'quadratic', 'exponential']}
    }
    
    # Advanced configuration
    config = BayesianTPEConfig(
        n_trials=40,
        coarse_grid_points=5,
        fine_grid_points=8,
        enable_early_stopping=True,
        early_stopping_patience=5,
        enable_convergence_detection=True,
        convergence_threshold=0.01,
        enable_parallel=True,
        max_workers=2,
        log_level='DEBUG'
    )
    
    # Optimize
    optimizer = BayesianTPEOptimizer(config)
    result = optimizer.optimize(objective_function, search_space)
    
    logger.info(f"✅ Best parameters: {result.best_params}")
    logger.info(f"📊 Best score: {result.best_score:.4f}")
    logger.info(f"🔧 Convergence info: {result.convergence_info}")
    
    return result


def example_4_convenience_function():
    """Example 4: Using convenience function."""
    logger.info("🎯 Example 4: Convenience function usage")
    
    def objective_function(params: Dict[str, Any], **kwargs) -> float:
        """Simple objective function."""
        return -(params['x'] - 1)**2 - (params['y'] - 2)**2
    
    # Create search space from bounds
    bounds = {
        'x': (-5.0, 5.0),
        'y': (-5.0, 5.0)
    }
    search_space = create_search_space_from_bounds(bounds)
    
    # Use convenience function
    result = optimize_with_bayesian_tpe(
        objective_function=objective_function,
        search_space=search_space,
        config=BayesianTPEConfig(n_trials=20)
    )
    
    logger.info(f"✅ Best parameters: {result.best_params}")
    logger.info(f"📊 Best score: {result.best_score:.4f}")
    
    return result


def example_5_error_handling():
    """Example 5: Error handling and robustness."""
    logger.info("🎯 Example 5: Error handling")
    
    def objective_function(params: Dict[str, Any], **kwargs) -> float:
        """Objective function that sometimes fails."""
        x = params['x']
        y = params['y']
        
        # Simulate occasional failures
        if np.random.random() < 0.1:  # 10% failure rate
            raise ValueError("Simulated failure")
        
        return -(x - 1)**2 - (y - 2)**2
    
    search_space = {
        'x': {'type': 'float', 'low': -5.0, 'high': 5.0},
        'y': {'type': 'float', 'low': -5.0, 'high': 5.0}
    }
    
    config = BayesianTPEConfig(
        n_trials=30,
        enable_progress_logging=True
    )
    
    optimizer = BayesianTPEOptimizer(config)
    result = optimizer.optimize(objective_function, search_space)
    
    logger.info(f"✅ Optimization completed despite errors")
    logger.info(f"📊 Best score: {result.best_score:.4f}")
    logger.info(f"🔧 Success: {result.success}")
    
    return result


def run_all_examples():
    """Run all examples."""
    logger.info("🚀 Running all Bayesian TPE examples")
    
    examples = [
        example_1_simple_function_optimization,
        example_2_machine_learning_optimization,
        example_3_advanced_configuration,
        example_4_convenience_function,
        example_5_error_handling
    ]
    
    results = []
    for i, example_func in enumerate(examples, 1):
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running Example {i}")
            logger.info(f"{'='*50}")
            
            result = example_func()
            results.append(result)
            
            logger.info(f"✅ Example {i} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Example {i} failed: {e}")
            results.append(None)
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("SUMMARY")
    logger.info(f"{'='*50}")
    
    successful = sum(1 for r in results if r is not None)
    logger.info(f"✅ Successful examples: {successful}/{len(examples)}")
    
    for i, result in enumerate(results, 1):
        if result is not None:
            logger.info(f"Example {i}: Score = {result.best_score:.4f}, Time = {result.optimization_time:.2f}s")
        else:
            logger.info(f"Example {i}: Failed")
    
    return results


def benchmark_performance():
    """Benchmark performance comparison."""
    logger.info("🏁 Performance benchmark")
    
    def objective_function(params: Dict[str, Any], **kwargs) -> float:
        """Benchmark objective function."""
        time.sleep(0.01)  # Simulate computation
        return -(params['x'] - 1)**2 - (params['y'] - 2)**2
    
    search_space = {
        'x': {'type': 'float', 'low': -5.0, 'high': 5.0},
        'y': {'type': 'float', 'low': -5.0, 'high': 5.0}
    }
    
    # Test different configurations
    configurations = [
        ("Grid + TPE", BayesianTPEConfig(n_trials=20, enable_grid_search=True)),
        ("TPE Only", BayesianTPEConfig(n_trials=20, enable_grid_search=False)),
        ("Parallel", BayesianTPEConfig(n_trials=20, enable_parallel=True, max_workers=4)),
        ("Sequential", BayesianTPEConfig(n_trials=20, enable_parallel=False))
    ]
    
    results = {}
    
    for name, config in configurations:
        logger.info(f"Testing {name}...")
        
        start_time = time.time()
        optimizer = BayesianTPEOptimizer(config)
        result = optimizer.optimize(objective_function, search_space)
        end_time = time.time()
        
        results[name] = {
            'score': result.best_score,
            'time': end_time - start_time,
            'success': result.success
        }
        
        logger.info(f"  Score: {result.best_score:.4f}")
        logger.info(f"  Time: {end_time - start_time:.2f}s")
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("BENCHMARK RESULTS")
    logger.info(f"{'='*50}")
    
    for name, metrics in results.items():
        logger.info(f"{name:15} | Score: {metrics['score']:8.4f} | Time: {metrics['time']:6.2f}s")
    
    return results


if __name__ == "__main__":
    # Run examples
    run_all_examples()
    
    # Run benchmark
    benchmark_performance()