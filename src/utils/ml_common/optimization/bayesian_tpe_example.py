"""
Bayesian TPE Optimizer Example Usage

This file demonstrates how to use the BayesianTPEOptimizer for various machine learning
models and optimization scenarios. It includes examples for XGBoost, LightGBM, Random Forest,
and neural networks, as well as custom evaluation functions and transfer learning.

Run this file to see the optimizer in action:
    python bayesian_tpe_example.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import logging

# Import the Bayesian TPE optimizer
from .bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    OptimizationConfig,
    TPEConfig,
    GridConfig,
    optimize_hyperparameters,
    create_optimization_config
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_xgb_model(params):
    """Create XGBoost model with given parameters."""
    try:
        from xgboost import XGBClassifier, XGBRegressor
        model_type = params.get('model_type', 'classifier')

        if model_type == 'classifier':
            return XGBClassifier(**params, verbosity=0)
        else:
            return XGBRegressor(**params, verbosity=0)
    except ImportError:
        logger.warning("XGBoost not available, using Random Forest as fallback")
        if params.get('model_type', 'classifier') == 'classifier':
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)


def create_lgb_model(params):
    """Create LightGBM model with given parameters."""
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        model_type = params.get('model_type', 'classifier')

        if model_type == 'classifier':
            return LGBMClassifier(**params, verbosity=-1)
        else:
            return LGBMRegressor(**params, verbosity=-1)
    except ImportError:
        logger.warning("LightGBM not available, using Random Forest as fallback")
        if params.get('model_type', 'classifier') == 'classifier':
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)


def create_nn_model(params):
    """Create neural network model with given parameters."""
    model_type = params.get('model_type', 'classifier')

    if model_type == 'classifier':
        return MLPClassifier(**params, random_state=42)
    else:
        return MLPRegressor(**params, random_state=42)


def custom_classification_evaluator(model, X, y):
    """Custom evaluation function for classification."""
    try:
        y_pred = model.predict(X)

        # Multi-objective scoring
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='macro')

        # Combined score (weighted average)
        return 0.7 * accuracy + 0.3 * f1
    except Exception as e:
        logger.warning(f"Custom evaluation failed: {e}")
        return 0.5


def custom_regression_evaluator(model, X, y):
    """Custom evaluation function for regression."""
    try:
        y_pred = model.predict(X)

        # Multi-objective scoring
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Convert MSE to score (lower is better for MSE, higher for R2)
        mse_score = max(0, 1 - mse / (np.var(y) + 1e-8))  # Normalize MSE
        combined_score = 0.5 * r2 + 0.5 * mse_score

        return combined_score
    except Exception as e:
        logger.warning(f"Custom evaluation failed: {e}")
        return 0.5


def example_classification_optimization():
    """Example: Classification optimization with XGBoost."""
    logger.info("=" * 60)
    logger.info("CLASSIFICATION OPTIMIZATION EXAMPLE")
    logger.info("=" * 60)

    # Generate sample classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define custom search space for XGBoost
    search_space = {
        'max_depth': {'type': 'int', 'low': 3, 'high': 8},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
        'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'reg_alpha': {'type': 'float', 'low': 0, 'high': 1},
        'reg_lambda': {'type': 'float', 'low': 0, 'high': 1},
        'model_type': {'type': 'categorical', 'choices': ['classifier']}
    }

    # Create optimization configuration
    config = create_optimization_config(
        n_trials=30,
        coarse_grid_points=4,
        fine_grid_points=6,
        enable_parallel=True,
        max_workers=2
    )

    # Create optimizer
    optimizer = BayesianTPEOptimizer(
        config=config,
        model_type='xgboost'
    )

    # Run optimization
    results = optimizer.optimize(
        model_factory=create_xgb_model,
        X=X_train,
        y=y_train,
        search_space=search_space,
        custom_evaluation_fn=lambda model, X, y: custom_classification_evaluator(model, X, y)
    )

    # Display results
    logger.info("Optimization Results:")
    logger.info(f"  Best Score: {results.best_score:.4f}")
    logger.info(f"  Best Stage: {results.best_stage}")
    logger.info(f"  Best Parameters: {results.best_params}")
    logger.info(f"  Total Trials: {results.n_trials_total}")
    logger.info(f"  Coarse Trials: {results.n_trials_coarse}")
    logger.info(f"  Fine Trials: {results.n_trials_fine}")
    logger.info(f"  TPE Trials: {results.n_trials_tpe}")
    logger.info(f"  Total Time: {results.optimization_time:.2f}s")

    # Test best model on test set
    best_model = create_xgb_model(results.best_params)
    best_model.fit(X_train, y_train)
    test_score = custom_classification_evaluator(best_model, X_test, y_test)
    logger.info(f"  Test Set Score: {test_score:.4f}")

    return results


def example_regression_optimization():
    """Example: Regression optimization with LightGBM."""
    logger.info("=" * 60)
    logger.info("REGRESSION OPTIMIZATION EXAMPLE")
    logger.info("=" * 60)

    # Generate sample regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        noise=0.1,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define custom search space for LightGBM
    search_space = {
        'num_leaves': {'type': 'int', 'low': 10, 'high': 50},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
        'n_estimators': {'type': 'int', 'low': 50, 'high': 200},
        'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'bagging_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0},
        'bagging_freq': {'type': 'int', 'low': 1, 'high': 5},
        'min_child_samples': {'type': 'int', 'low': 5, 'high': 20},
        'reg_alpha': {'type': 'float', 'low': 0, 'high': 1},
        'reg_lambda': {'type': 'float', 'low': 0, 'high': 1},
        'model_type': {'type': 'categorical', 'choices': ['regressor']}
    }

    # Create optimization configuration with custom validation
    config = OptimizationConfig(
        tpe_config=TPEConfig(n_trials=25, enable_parallel=False),
        grid_config=GridConfig(coarse_grid_points=3, fine_grid_points=5),
        validation_config={
            'cv_folds': 3,
            'scoring': 'neg_mean_squared_error',
            'test_size': 0.2,
            'random_state': 42
        }
    )

    # Run optimization
    results = optimize_hyperparameters(
        model_factory=create_lgb_model,
        X=X_train,
        y=y_train,
        search_space=search_space,
        model_type='lightgbm',
        config=config,
        custom_evaluation_fn=lambda model, X, y: custom_regression_evaluator(model, X, y)
    )

    # Display results
    logger.info("Optimization Results:")
    logger.info(f"  Best Score: {results.best_score:.4f}")
    logger.info(f"  Best Stage: {results.best_stage}")
    logger.info(f"  Best Parameters: {results.best_params}")
    logger.info(f"  Total Trials: {results.n_trials_total}")
    logger.info(f"  Total Time: {results.optimization_time:.2f}s")

    # Test best model on test set
    best_model = create_lgb_model(results.best_params)
    best_model.fit(X_train, y_train)
    test_score = custom_regression_evaluator(best_model, X_test, y_test)
    logger.info(f"  Test Set Score: {test_score:.4f}")

    return results


def example_neural_network_optimization():
    """Example: Neural network optimization."""
    logger.info("=" * 60)
    logger.info("NEURAL NETWORK OPTIMIZATION EXAMPLE")
    logger.info("=" * 60)

    # Generate sample data
    X, y = make_classification(
        n_samples=800,
        n_features=10,
        n_informative=8,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define search space for neural networks
    search_space = {
        'hidden_layer_sizes': {'type': 'categorical',
                              'choices': [(50,), (100,), (50, 25), (100, 50)]},
        'activation': {'type': 'categorical',
                      'choices': ['relu', 'tanh', 'logistic']},
        'learning_rate': {'type': 'categorical',
                         'choices': ['constant', 'adaptive']},
        'learning_rate_init': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
        'max_iter': {'type': 'int', 'low': 100, 'high': 500},
        'batch_size': {'type': 'int', 'low': 16, 'high': 64},
        'alpha': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True},
        'model_type': {'type': 'categorical', 'choices': ['classifier']}
    }

    # Create optimization configuration
    config = create_optimization_config(
        n_trials=20,
        coarse_grid_points=3,
        fine_grid_points=4,
        enable_parallel=False  # Neural networks can be memory intensive
    )

    # Run optimization
    results = optimize_hyperparameters(
        model_factory=create_nn_model,
        X=X_train,
        y=y_train,
        search_space=search_space,
        model_type='neural_network',
        config=config
    )

    # Display results
    logger.info("Optimization Results:")
    logger.info(f"  Best Score: {results.best_score:.4f}")
    logger.info(f"  Best Stage: {results.best_stage}")
    logger.info(f"  Best Parameters: {results.best_params}")
    logger.info(f"  Total Trials: {results.n_trials_total}")
    logger.info(f"  Total Time: {results.optimization_time:.2f}s")

    return results


def example_transfer_learning():
    """Example: Transfer learning between similar datasets."""
    logger.info("=" * 60)
    logger.info("TRANSFER LEARNING EXAMPLE")
    logger.info("=" * 60)

    # Generate two similar datasets
    X1, y1 = make_classification(
        n_samples=800,
        n_features=15,
        n_informative=10,
        n_classes=3,
        random_state=42
    )

    X2, y2 = make_classification(
        n_samples=600,
        n_features=15,
        n_informative=10,
        n_classes=3,
        random_state=123  # Different seed for different data
    )

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

    # First optimization on dataset 1
    logger.info("Running initial optimization on dataset 1...")
    config = create_optimization_config(n_trials=15, coarse_grid_points=3, fine_grid_points=4)

    optimizer1 = BayesianTPEOptimizer(config=config, model_type='xgboost')
    results1 = optimizer1.optimize(create_xgb_model, X1_train, y1_train)

    # Transfer learning on dataset 2
    logger.info("Running transfer learning optimization on dataset 2...")

    optimizer2 = BayesianTPEOptimizer(
        config=OptimizationConfig(
            tpe_config=TPEConfig(n_trials=15),
            grid_config=GridConfig(coarse_enabled=False, fine_enabled=False),  # Skip grid stages for transfer
            transfer_learning_threshold=0.7
        ),
        model_type='xgboost'
    )

    results2 = optimizer2.optimize(
        model_factory=create_xgb_model,
        X=X2_train,
        y=y2_train,
        transfer_learning_data={
            'best_params': results1.best_params,
            'best_score': results1.best_score,
            'n_samples': len(X1),
            'n_features': X1.shape[1],
            'n_classes': len(np.unique(y1))
        }
    )

    # Compare results
    logger.info("Transfer Learning Results:")
    logger.info(f"  Dataset 1 Score: {results1.best_score:.4f}")
    logger.info(f"  Dataset 2 Score (Transfer): {results2.best_score:.4f}")
    logger.info(f"  Dataset 2 Stage: {results2.best_stage}")

    return results1, results2


def example_custom_evaluation_function():
    """Example: Using completely custom evaluation function."""
    logger.info("=" * 60)
    logger.info("CUSTOM EVALUATION FUNCTION EXAMPLE")
    logger.info("=" * 60)

    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define search space
    search_space = {
        'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
        'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 5}
    }

    def business_metric_evaluator(model, X, y):
        """
        Custom business metric that combines accuracy with model complexity.
        """
        try:
            # Standard accuracy
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)

            # Complexity penalty (fewer estimators = simpler model = better)
            n_estimators = getattr(model, 'n_estimators', 50)
            complexity_penalty = n_estimators / 100.0  # Penalty for complex models

            # Business score: accuracy minus complexity penalty
            business_score = accuracy - complexity_penalty * 0.1

            return business_score

        except Exception as e:
            logger.warning(f"Business metric evaluation failed: {e}")
            return 0.5

    # Run optimization with custom evaluator
    results = optimize_hyperparameters(
        model_factory=lambda params: RandomForestClassifier(**params, random_state=42),
        X=X_train,
        y=y_train,
        search_space=search_space,
        model_type='random_forest',
        config=create_optimization_config(n_trials=10, coarse_grid_points=2, fine_grid_points=3),
        custom_evaluation_fn=business_metric_evaluator
    )

    logger.info("Custom Evaluation Results:")
    logger.info(f"  Business Score: {results.best_score:.4f}")
    logger.info(f"  Best Parameters: {results.best_params}")

    # Test on test set
    best_model = RandomForestClassifier(**results.best_params, random_state=42)
    best_model.fit(X_train, y_train)
    test_business_score = business_metric_evaluator(best_model, X_test, y_test)
    logger.info(f"  Test Business Score: {test_business_score:.4f}")

    return results


def main():
    """Run all examples."""
    logger.info("🚀 Starting Bayesian TPE Optimizer Examples")
    logger.info("=" * 80)

    try:
        # Run examples
        results = []

        # Classification example
        results.append(example_classification_optimization())

        # Regression example
        results.append(example_regression_optimization())

        # Neural network example
        results.append(example_neural_network_optimization())

        # Transfer learning example
        results.extend(example_transfer_learning())

        # Custom evaluation example
        results.append(example_custom_evaluation_function())

        logger.info("=" * 80)
        logger.info("✅ All examples completed successfully!")
        logger.info(f"   Completed {len(results)} optimization runs")

    except Exception as e:
        logger.error(f"❌ Examples failed: {e}")
        raise


if __name__ == "__main__":
    main()