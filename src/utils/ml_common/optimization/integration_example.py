"""
Integration Example: Bayesian TPE Optimizer

This example demonstrates how to integrate the new Bayesian TPE optimizer
with your existing codebase, showing how it automatically calls your grid utils.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

# Import the new Bayesian TPE optimizer
from .bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    BayesianTPEConfig,
    optimize_with_bayesian_tpe,
    create_search_space_from_bounds
)

# Import existing utilities (these are automatically used by the optimizer)
from .grid_utils import build_coarse_grid_from_search_space, build_fine_grid_around_best
from .hpo_utils import HyperparameterOptimization
from src.utils.nas_tas.hierarchical_hpo import HierarchicalHPO, HierarchicalHPOConfig, HPOPhaseConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_simple_integration():
    """Example 1: Simple integration with existing grid utils."""
    logger.info("🎯 Example 1: Simple integration")
    
    def objective_function(params: Dict[str, Any], **kwargs) -> float:
        """Simple objective function."""
        x, y = params['x'], params['y']
        return -(x - 1)**2 - (y - 2)**2
    
    # Define search space
    search_space = {
        'x': {'type': 'float', 'low': -5.0, 'high': 5.0},
        'y': {'type': 'float', 'low': -5.0, 'high': 5.0}
    }
    
    # Configure optimizer to use your existing grid utils
    config = BayesianTPEConfig(
        n_trials=30,
        coarse_grid_points=5,  # Uses build_coarse_grid_from_search_space
        fine_grid_points=8,    # Uses build_fine_grid_around_best
        enable_grid_search=True,
        log_level='INFO'
    )
    
    # Create optimizer
    optimizer = BayesianTPEOptimizer(config)
    
    # Optimize (automatically calls your grid utils)
    result = optimizer.optimize(objective_function, search_space)
    
    logger.info(f"✅ Best parameters: {result.best_params}")
    logger.info(f"📊 Best score: {result.best_score:.4f}")
    logger.info(f"⏱️ Time: {result.optimization_time:.2f}s")
    logger.info(f"🔧 Grid search used: {result.convergence_info['grid_search_used']}")
    
    return result


def example_2_ml_hyperparameter_optimization():
    """Example 2: ML hyperparameter optimization with data."""
    logger.info("🎯 Example 2: ML hyperparameter optimization")
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)
    
    def ml_objective(params: Dict[str, Any], X: np.ndarray, y: np.ndarray, **kwargs) -> float:
        """ML objective function."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        
        model = RandomForestRegressor(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_samples_split=int(params['min_samples_split']),
            random_state=42
        )
        
        scores = cross_val_score(model, X, y, cv=3, scoring='r2')
        return np.mean(scores)
    
    # Create search space from bounds
    bounds = {
        'n_estimators': (10, 100),
        'max_depth': (3, 20),
        'min_samples_split': (2, 10)
    }
    
    param_types = {
        'n_estimators': 'int',
        'max_depth': 'int',
        'min_samples_split': 'int'
    }
    
    search_space = create_search_space_from_bounds(bounds, param_types)
    
    # Configure for ML optimization
    config = BayesianTPEConfig(
        n_trials=25,
        coarse_grid_points=4,
        fine_grid_points=6,
        enable_grid_search=True,
        enable_parallel=True,
        max_workers=2,
        backend='optuna'
    )
    
    # Optimize
    result = optimize_with_bayesian_tpe(
        objective_function=ml_objective,
        search_space=search_space,
        config=config,
        X=X,
        y=y
    )
    
    logger.info(f"✅ Best ML parameters: {result.best_params}")
    logger.info(f"📊 Best R² score: {result.best_score:.4f}")
    
    return result


def example_3_advanced_configuration():
    """Example 3: Advanced configuration with monitoring."""
    logger.info("🎯 Example 3: Advanced configuration")
    
    def complex_objective(params: Dict[str, Any], **kwargs) -> float:
        """Complex objective function."""
        x1, x2, x3 = params['x1'], params['x2'], params['x3']
        method = params['method']
        
        if method == 'linear':
            return x1 + x2 + x3
        elif method == 'quadratic':
            return -(x1**2 + x2**2 + x3**2)
        else:  # exponential
            return np.exp(-(x1**2 + x2**2 + x3**2))
    
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
        enable_grid_search=True,
        enable_early_stopping=True,
        early_stopping_patience=5,
        enable_convergence_detection=True,
        convergence_threshold=0.01,
        enable_parallel=True,
        max_workers=2,
        enable_performance_monitoring=True,
        monitor_memory=True,
        monitor_time=True,
        log_level='DEBUG'
    )
    
    optimizer = BayesianTPEOptimizer(config)
    result = optimizer.optimize(complex_objective, search_space)
    
    logger.info(f"✅ Best parameters: {result.best_params}")
    logger.info(f"📊 Best score: {result.best_score:.4f}")
    logger.info(f"🔧 Convergence info: {result.convergence_info}")
    
    # Show optimization history
    for entry in result.optimization_history:
        logger.info(f"   → {entry['stage']}: {entry['best_score']:.4f}")
    
    return result


def example_4_integration_with_existing_hpo():
    """Example 4: Integration with existing HPO utilities."""
    logger.info("🎯 Example 4: Integration with existing HPO")
    
    def objective_function(params: Dict[str, Any], **kwargs) -> float:
        """Objective function."""
        return -(params['x'] - 1)**2 - (params['y'] - 2)**2
    
    search_space = {
        'x': {'type': 'float', 'low': -5.0, 'high': 5.0},
        'y': {'type': 'float', 'low': -5.0, 'high': 5.0}
    }
    
    # Use existing HPO utilities alongside new Bayesian TPE
    hpo_config = {
        'enable_parallel': True,
        'max_workers': 2,
        'use_nonlinear_optimization': True
    }
    
    # Create existing HPO instance
    existing_hpo = HyperparameterOptimization(hpo_config)
    
    # Use new Bayesian TPE optimizer
    tpe_config = BayesianTPEConfig(
        n_trials=20,
        enable_grid_search=True,
        coarse_grid_points=4,
        fine_grid_points=6
    )
    
    tpe_optimizer = BayesianTPEOptimizer(tpe_config)
    
    # Compare results
    logger.info("Running existing HPO...")
    # Note: This would require implementing the comparison
    # existing_result = existing_hpo.bayesian_optimization(...)
    
    logger.info("Running new Bayesian TPE...")
    tpe_result = tpe_optimizer.optimize(objective_function, search_space)
    
    logger.info(f"✅ TPE result: {tpe_result.best_score:.4f}")
    logger.info(f"🔧 Grid search used: {tpe_result.convergence_info['grid_search_used']}")
    
    return tpe_result


def example_5_error_handling_and_robustness():
    """Example 5: Error handling and robustness."""
    logger.info("🎯 Example 5: Error handling and robustness")
    
    def risky_objective(params: Dict[str, Any], **kwargs) -> float:
        """Objective function that sometimes fails."""
        x, y = params['x'], params['y']
        
        # Simulate occasional failures
        if np.random.random() < 0.15:  # 15% failure rate
            raise ValueError("Simulated evaluation failure")
        
        return -(x - 1)**2 - (y - 2)**2
    
    search_space = {
        'x': {'type': 'float', 'low': -5.0, 'high': 5.0},
        'y': {'type': 'float', 'low': -5.0, 'high': 5.0}
    }
    
    config = BayesianTPEConfig(
        n_trials=30,
        enable_grid_search=True,
        enable_progress_logging=True,
        log_level='INFO'
    )
    
    optimizer = BayesianTPEOptimizer(config)
    result = optimizer.optimize(risky_objective, search_space)
    
    logger.info(f"✅ Optimization completed despite errors")
    logger.info(f"📊 Best score: {result.best_score:.4f}")
    logger.info(f"🔧 Success: {result.success}")
    
    if result.error_message:
        logger.info(f"⚠️ Error message: {result.error_message}")
    
    return result


def run_integration_examples():
    """Run all integration examples."""
    logger.info("🚀 Running Bayesian TPE integration examples")
    
    examples = [
        example_1_simple_integration,
        example_2_ml_hyperparameter_optimization,
        example_3_advanced_configuration,
        example_4_integration_with_existing_hpo,
        example_5_error_handling_and_robustness
    ]
    
    results = []
    
    for i, example_func in enumerate(examples, 1):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running Integration Example {i}")
            logger.info(f"{'='*60}")
            
            result = example_func()
            results.append(result)
            
            logger.info(f"✅ Integration Example {i} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Integration Example {i} failed: {e}")
            results.append(None)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("INTEGRATION SUMMARY")
    logger.info(f"{'='*60}")
    
    successful = sum(1 for r in results if r is not None)
    logger.info(f"✅ Successful examples: {successful}/{len(examples)}")
    
    for i, result in enumerate(results, 1):
        if result is not None:
            logger.info(f"Example {i}: Score = {result.best_score:.4f}, Time = {result.optimization_time:.2f}s")
        else:
            logger.info(f"Example {i}: Failed")
    
    return results


if __name__ == "__main__":
    # Run integration examples
    run_integration_examples()