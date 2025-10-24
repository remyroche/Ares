"""
Enhanced HPO Example and Integration Script

This module demonstrates how to use the enhanced HPO system with all
the new features: multi-objective optimization, early stopping, warm starting,
and concurrent model optimization.

Enhancement: Complete example and integration
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Callable
import logging
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import time

# Import enhanced HPO components
from .enhanced_hpo_engine import (
    EnhancedHPOEngine, EnhancedHPOConfig, create_enhanced_hpo_engine,
    create_multi_model_optimization_config
)
from .multi_objective_optimizer import (
    create_accuracy_efficiency_objectives, create_performance_robustness_objectives
)
from .enhanced_early_stopping_integration import create_early_stopping_integration
from .warm_starting_system import create_warm_start_manager
from .validation import HPOConfig

logger = logging.getLogger(__name__)


def create_sample_data(n_samples: int = 1000, n_features: int = 20, 
                      problem_type: str = 'classification') -> tuple:
    """Create sample dataset for demonstration."""
    if problem_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features//2,
            n_redundant=n_features//4,
            random_state=42
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=42
        )
    
    return X, y


def create_model_factory(model_class, **default_params):
    """Create model factory function."""
    def factory(**params):
        all_params = {**default_params, **params}
        return model_class(**all_params)
    return factory


def create_search_spaces() -> Dict[str, Dict[str, Any]]:
    """Create search spaces for different model types."""
    return {
        'random_forest': {
            'n_estimators': {'type': 'int', 'low': 10, 'high': 200},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]}
        },
        'gradient_boosting': {
            'n_estimators': {'type': 'int', 'low': 10, 'high': 200},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0}
        },
        'ridge': {
            'alpha': {'type': 'float', 'low': 0.001, 'high': 100.0, 'log': True}
        },
        'lasso': {
            'alpha': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True}
        }
    }


def example_single_objective_optimization():
    """Example of single-objective optimization with enhanced features."""
    print("=" * 60)
    print("Single-Objective Optimization with Enhanced Features")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=1000, n_features=20, problem_type='classification')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model factory
    model_factory = create_model_factory(RandomForestClassifier)
    search_space = create_search_spaces()['random_forest']
    
    # Create enhanced HPO engine
    hpo_engine = create_enhanced_hpo_engine(
        enable_early_stopping=True,
        enable_warm_start=True,
        enable_concurrent=False
    )
    
    # Optimize
    print("Starting optimization...")
    start_time = time.time()
    
    result = hpo_engine.optimize_single_model(
        model_factory=model_factory,
        X=X_train,
        y=y_train,
        search_space=search_space,
        model_name='random_forest',
        use_warm_start=True,
        use_early_stopping=True
    )
    
    optimization_time = time.time() - start_time
    
    # Display results
    print(f"\nOptimization completed in {optimization_time:.2f}s")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Number of trials: {result.n_trials}")
    print(f"Best parameters: {result.best_params}")
    
    # Test on holdout set
    best_model = model_factory(**result.best_params)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    return result


def example_multi_objective_optimization():
    """Example of multi-objective optimization."""
    print("\n" + "=" * 60)
    print("Multi-Objective Optimization")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=1000, n_features=20, problem_type='classification')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model factory
    model_factory = create_model_factory(RandomForestClassifier)
    search_space = create_search_spaces()['random_forest']
    
    # Create enhanced HPO engine with multi-objective
    hpo_engine = create_enhanced_hpo_engine(
        enable_multi_objective=True,
        enable_early_stopping=True,
        enable_warm_start=True
    )
    
    # Add custom objectives
    def accuracy_objective(params, model, X, y, **kwargs):
        """Accuracy objective."""
        try:
            model.fit(X, y)
            if hasattr(model, 'score'):
                return model.score(X, y)
            return 0.0
        except:
            return 0.0
    
    def efficiency_objective(params, model, X, y, **kwargs):
        """Efficiency objective (inverse of training time)."""
        try:
            start_time = time.time()
            model.fit(X, y)
            training_time = time.time() - start_time
            return 1.0 / (training_time + 1e-6)
        except:
            return 0.0
    
    # Add objectives to multi-objective optimizer
    if hpo_engine.multi_objective_optimizer:
        hpo_engine.multi_objective_optimizer.add_objective('accuracy', accuracy_objective, direction='maximize')
        hpo_engine.multi_objective_optimizer.add_objective('efficiency', efficiency_objective, direction='maximize')
    
    # Optimize
    print("Starting multi-objective optimization...")
    start_time = time.time()
    
    result = hpo_engine.optimize_single_model(
        model_factory=model_factory,
        X=X_train,
        y=y_train,
        search_space=search_space,
        model_name='random_forest_multi',
        use_warm_start=True,
        use_early_stopping=True
    )
    
    optimization_time = time.time() - start_time
    
    # Display results
    print(f"\nMulti-objective optimization completed in {optimization_time:.2f}s")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Number of trials: {result.n_trials}")
    print(f"Best parameters: {result.best_params}")
    
    # Display Pareto front information
    if 'pareto_front' in result.metadata:
        pareto_front = result.metadata['pareto_front']
        print(f"Pareto front size: {len(pareto_front)}")
        print(f"Diverse solutions: {len(result.metadata.get('diverse_solutions', []))}")
    
    return result


def example_concurrent_optimization():
    """Example of concurrent multi-model optimization."""
    print("\n" + "=" * 60)
    print("Concurrent Multi-Model Optimization")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=1000, n_features=20, problem_type='classification')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model configurations
    model_configs = [
        {
            'model_name': 'random_forest',
            'model_factory': create_model_factory(RandomForestClassifier),
            'use_warm_start': True,
            'use_early_stopping': True
        },
        {
            'model_name': 'gradient_boosting',
            'model_factory': create_model_factory(GradientBoostingRegressor),
            'use_warm_start': True,
            'use_early_stopping': True
        },
        {
            'model_name': 'ridge',
            'model_factory': create_model_factory(Ridge),
            'use_warm_start': True,
            'use_early_stopping': True
        }
    ]
    
    # Create search spaces
    search_spaces = [
        create_search_spaces()['random_forest'],
        create_search_spaces()['gradient_boosting'],
        create_search_spaces()['ridge']
    ]
    
    # Create enhanced HPO engine with concurrent optimization
    hpo_engine = create_enhanced_hpo_engine(
        enable_early_stopping=True,
        enable_warm_start=True,
        enable_concurrent=True,
        max_concurrent_models=3
    )
    
    # Optimize concurrently
    print("Starting concurrent optimization...")
    start_time = time.time()
    
    results = hpo_engine.optimize_multiple_models(
        model_configs=model_configs,
        X=X_train,
        y=y_train,
        search_spaces=search_spaces,
        use_concurrent=True
    )
    
    optimization_time = time.time() - start_time
    
    # Display results
    print(f"\nConcurrent optimization completed in {optimization_time:.2f}s")
    print(f"Successful optimizations: {len(results)}")
    
    for i, result in enumerate(results):
        if result:
            print(f"\nModel {i+1} ({model_configs[i]['model_name']}):")
            print(f"  Best score: {result.best_score:.4f}")
            print(f"  Number of trials: {result.n_trials}")
            print(f"  Optimization time: {result.optimization_time:.2f}s")
    
    return results


def example_warm_starting():
    """Example of warm starting from previous optimizations."""
    print("\n" + "=" * 60)
    print("Warm Starting Example")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=1000, n_features=20, problem_type='classification')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model factory
    model_factory = create_model_factory(RandomForestClassifier)
    search_space = create_search_spaces()['random_forest']
    
    # First optimization (will be saved as warm start data)
    print("First optimization (creating warm start data)...")
    hpo_engine1 = create_enhanced_hpo_engine(
        enable_early_stopping=True,
        enable_warm_start=True
    )
    
    result1 = hpo_engine1.optimize_single_model(
        model_factory=model_factory,
        X=X_train,
        y=y_train,
        search_space=search_space,
        model_name='random_forest_warm_start',
        use_warm_start=False,  # No warm start for first run
        use_early_stopping=True
    )
    
    print(f"First optimization completed: {result1.best_score:.4f}")
    
    # Second optimization (using warm start)
    print("\nSecond optimization (using warm start)...")
    hpo_engine2 = create_enhanced_hpo_engine(
        enable_early_stopping=True,
        enable_warm_start=True
    )
    
    # Copy warm start data from first engine
    if hpo_engine1.warm_start_manager and hpo_engine2.warm_start_manager:
        for data in hpo_engine1.warm_start_manager.warm_start_data:
            hpo_engine2.warm_start_manager.add_warm_start_data(data)
    
    result2 = hpo_engine2.optimize_single_model(
        model_factory=model_factory,
        X=X_train,
        y=y_train,
        search_space=search_space,
        model_name='random_forest_warm_start_2',
        use_warm_start=True,  # Use warm start
        use_early_stopping=True
    )
    
    print(f"Second optimization completed: {result2.best_score:.4f}")
    print(f"Improvement: {result2.best_score - result1.best_score:.4f}")
    
    return result1, result2


def example_early_stopping():
    """Example of early stopping in action."""
    print("\n" + "=" * 60)
    print("Early Stopping Example")
    print("=" * 60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=500, n_features=10, problem_type='classification')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create model factory
    model_factory = create_model_factory(RandomForestClassifier)
    search_space = create_search_spaces()['random_forest']
    
    # Create HPO engine with aggressive early stopping
    hpo_engine = create_enhanced_hpo_engine(
        enable_early_stopping=True,
        enable_warm_start=False
    )
    
    # Configure aggressive early stopping
    if hpo_engine.early_stopping_integration:
        hpo_engine.early_stopping_integration.config.early_stopping_patience = 3
        hpo_engine.early_stopping_integration.config.early_stopping_threshold = 0.001
    
    # Optimize
    print("Starting optimization with early stopping...")
    start_time = time.time()
    
    result = hpo_engine.optimize_single_model(
        model_factory=model_factory,
        X=X_train,
        y=y_train,
        search_space=search_space,
        model_name='random_forest_early_stop',
        use_warm_start=False,
        use_early_stopping=True
    )
    
    optimization_time = time.time() - start_time
    
    # Display results
    print(f"\nOptimization completed in {optimization_time:.2f}s")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Number of trials: {result.n_trials}")
    print(f"Early stopped: {result.metadata.get('early_stopping', {}).get('early_stopped', False)}")
    
    return result


def run_comprehensive_example():
    """Run comprehensive example with all features."""
    print("Enhanced HPO System - Comprehensive Example")
    print("=" * 80)
    
    try:
        # Single-objective optimization
        result1 = example_single_objective_optimization()
        
        # Multi-objective optimization
        result2 = example_multi_objective_optimization()
        
        # Concurrent optimization
        results3 = example_concurrent_optimization()
        
        # Warm starting
        result4a, result4b = example_warm_starting()
        
        # Early stopping
        result5 = example_early_stopping()
        
        print("\n" + "=" * 80)
        print("Comprehensive Example Completed Successfully!")
        print("=" * 80)
        
        # Summary
        print(f"\nSummary:")
        print(f"- Single-objective optimization: {result1.best_score:.4f}")
        print(f"- Multi-objective optimization: {result2.best_score:.4f}")
        print(f"- Concurrent optimization: {len(results3)} models")
        print(f"- Warm starting improvement: {result4b.best_score - result4a.best_score:.4f}")
        print(f"- Early stopping efficiency: {result5.n_trials} trials")
        
    except Exception as e:
        print(f"Error in comprehensive example: {e}")
        logger.error(f"Comprehensive example failed: {e}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run comprehensive example
    run_comprehensive_example()