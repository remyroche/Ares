"""
Example: Hierarchical Parameter Optimization

This script demonstrates how to use the HierarchicalParameterOptimizer
to efficiently tune hyperparameters for a LightGBM model.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

from hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    create_param_group,
    OptimizationStage,
    default_objective_function
)


def main():
    """Run example hierarchical optimization."""
    
    print("=" * 80)
    print("Hierarchical Parameter Optimization Example")
    print("=" * 80)
    print()
    
    # Generate synthetic dataset
    print("1. Generating synthetic dataset...")
    X, y = make_regression(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        noise=10.0,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print()
    
    # Define parameter groups
    print("2. Defining parameter groups...")
    param_groups = [
        # Group 1: Model structure (highest priority)
        create_param_group(
            name="structure",
            params={
                "n_estimators": {"type": "int", "low": 50, "high": 300},
                "num_leaves": {"type": "int", "low": 20, "high": 100},
                "max_depth": {"type": "int", "low": 3, "high": 10}
            },
            priority=1,
            description="Core model structure parameters"
        ),
        
        # Group 2: Learning parameters
        create_param_group(
            name="learning",
            params={
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "min_child_samples": {"type": "int", "low": 10, "high": 100}
            },
            priority=2,
            depends_on=["structure"],
            description="Learning rate and related parameters"
        ),
        
        # Group 3: Regularization (fine-tuning)
        create_param_group(
            name="regularization",
            params={
                "reg_alpha": {"type": "float", "low": 0.0, "high": 1.0},
                "reg_lambda": {"type": "float", "low": 0.0, "high": 1.0},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0}
            },
            priority=3,
            depends_on=["structure", "learning"],
            description="Regularization parameters"
        )
    ]
    
    print(f"   Created {len(param_groups)} parameter groups:")
    for group in param_groups:
        print(f"   - {group.name}: {len(group.params)} parameters (priority={group.priority})")
    print()
    
    # Create model
    print("3. Creating LightGBM model...")
    model = LGBMRegressor(random_state=42, verbose=-1)
    print()
    
    # Create optimizer
    print("4. Creating hierarchical optimizer...")
    optimizer = HierarchicalParameterOptimizer(
        param_groups=param_groups,
        objective_func=default_objective_function,
        stages=[
            OptimizationStage.COARSE_GRID,
            OptimizationStage.FINE_GRID,
            OptimizationStage.TPE
        ],
        cv_folds=3,  # Use 3 folds for speed in this example
        scoring_metric='neg_mean_squared_error',
        direction='maximize',
        enable_final_refinement=True,
        final_refinement_trials=30,
        cache_dir="./optimization_cache",
        random_state=42,
        verbose=True
    )
    print()
    
    # Run optimization
    print("5. Running hierarchical optimization...")
    print("   This will optimize each parameter group sequentially.")
    print()
    
    result = optimizer.optimize(
        X_train=X_train,
        y_train=y_train,
        model=model
    )
    
    print()
    print("=" * 80)
    print("Optimization Complete!")
    print("=" * 80)
    print()
    
    # Display results
    print("6. Results Summary:")
    print(f"   Best Score: {result.best_score:.6f}")
    print(f"   Total Time: {result.total_time:.2f}s")
    print(f"   Total Trials: {result.total_trials}")
    print()
    
    print("   Best Parameters:")
    for param_name, param_value in sorted(result.best_params.items()):
        print(f"   - {param_name}: {param_value}")
    print()
    
    print("   Group-wise Results:")
    for i, group_result in enumerate(result.group_results):
        print(f"   {i+1}. {group_result.group_name}:")
        print(f"      Score: {group_result.best_score:.6f}")
        print(f"      Trials: {group_result.n_trials}")
        print(f"      Time: {group_result.optimization_time:.2f}s")
        print(f"      Best params: {group_result.best_params}")
    print()
    
    if result.final_refinement_result:
        print("   Final Refinement:")
        print(f"      Score: {result.final_refinement_result.best_score:.6f}")
        print(f"      Trials: {result.final_refinement_result.n_trials}")
        print(f"      Time: {result.final_refinement_result.optimization_time:.2f}s")
        print()
    
    # Train final model with optimized parameters
    print("7. Training final model with optimized parameters...")
    model.set_params(**result.best_params)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"   Test MSE: {test_mse:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    print()
    
    # Compare with default parameters
    print("8. Comparing with default parameters...")
    default_model = LGBMRegressor(random_state=42, verbose=-1)
    default_model.fit(X_train, y_train)
    y_pred_default = default_model.predict(X_test)
    default_mse = mean_squared_error(y_test, y_pred_default)
    default_r2 = r2_score(y_test, y_pred_default)
    
    print(f"   Default MSE: {default_mse:.4f}")
    print(f"   Default R²: {default_r2:.4f}")
    print()
    
    improvement_mse = ((default_mse - test_mse) / default_mse) * 100
    improvement_r2 = ((test_r2 - default_r2) / abs(default_r2)) * 100
    
    print(f"   Improvement in MSE: {improvement_mse:.2f}%")
    print(f"   Improvement in R²: {improvement_r2:.2f}%")
    print()
    
    print("=" * 80)
    print("Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
