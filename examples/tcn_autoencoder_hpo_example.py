"""
Example: Hyperparameter Optimization for Autoencoder + TCN using Hierarchical Optimizer.

This script demonstrates how to use the hierarchical parameter optimizer to efficiently
tune the Autoencoder + TCN architecture for analyst/tactician models.

The optimizer uses a staged approach:
1. Round 1: Exploration
   - Autoencoder structure (latent_dim, hidden_dim)
   - TCN structure (num_filters, num_layers, kernel_size, dilation_base)
   - Learning rates
   - Regularization
   - Training parameters

2. Round 2: Refinement
   - Narrow search around best parameters from Round 1
   - Capture parameter interactions across groups

3. Final Refinement: Joint optimization of all parameters
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.models_training.core.tcn_autoencoder_hpo import (
    AutoencoderTCNHPO,
    optimize_analyst_autoencoder_tcn,
    optimize_tactician_autoencoder_tcn
)
from src.models.causal_dilated_tcn import CausalTCNConfig, CausalDilatedTCNModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_synthetic_market_data(n_samples=2000, n_features=120, role="analyst"):
    """
    Generate synthetic market-like data for testing HPO.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        role: "analyst" or "tactician"
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    logger.info(f"📊 Generating synthetic {role} data...")
    logger.info(f"   Samples: {n_samples}, Features: {n_features}")
    
    # Generate correlated features (simulating market indicators)
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    # Add temporal structure
    for i in range(1, n_features):
        X[:, i] = 0.6 * X[:, i-1] + 0.4 * X[:, i]
    
    # Generate targets based on feature patterns
    if role == "analyst":
        # Green light prediction (binary)
        signal = X[:, :10].mean(axis=1) + 0.3 * X[:, 20:30].std(axis=1)
        y = (signal > 0).astype(int)
    else:  # tactician
        # Entry timing (binary)
        signal = X[:, :5].mean(axis=1) - 0.2 * X[:, 10:20].std(axis=1)
        y = (signal > 0).astype(int)
    
    # Split into train/val/test
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]
    
    logger.info(f"✅ Data split:")
    logger.info(f"   Train: {len(X_train)} ({np.mean(y_train):.2%} positive)")
    logger.info(f"   Val: {len(X_val)} ({np.mean(y_val):.2%} positive)")
    logger.info(f"   Test: {len(X_test)} ({np.mean(y_test):.2%} positive)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def example_1_basic_analyst_hpo():
    """Example 1: Basic HPO for Analyst model."""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 1: Basic Hyperparameter Optimization for Analyst")
    logger.info("="*80 + "\n")
    
    # Generate data
    X_train, y_train, X_val, y_val, X_test, y_test = generate_synthetic_market_data(
        n_samples=2000,
        n_features=120,
        role="analyst"
    )
    
    # Run optimization (uses convenience function)
    logger.info("\n🎯 Starting hyperparameter optimization...")
    logger.info("   This will optimize 5 parameter groups across 2 rounds\n")
    
    best_params, best_score = optimize_analyst_autoencoder_tcn(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        metric="accuracy",
        n_rounds=2,
        save_results=True
    )
    
    logger.info("\n" + "="*80)
    logger.info("✅ Optimization Complete!")
    logger.info("="*80)
    logger.info(f"Best validation accuracy: {best_score:.4f}")
    logger.info("\nBest parameters:")
    for param, value in best_params.items():
        logger.info(f"   {param}: {value}")
    
    # Train final model with best parameters
    logger.info("\n🏋️ Training final model with optimized parameters...")
    
    config = CausalTCNConfig(
        use_autoencoder=True,
        latent_dim=best_params['latent_dim'],
        num_filters=best_params['num_filters'],
        num_layers=best_params['num_layers'],
        kernel_size=best_params['kernel_size'],
        dilation_base=best_params['dilation_base'],
        learning_rate=best_params['tcn_learning_rate'],
        dropout=best_params['tcn_dropout'],
        batch_size=best_params['batch_size'],
        epochs=best_params['tcn_epochs'],
        early_stopping_patience=best_params['early_stopping_patience'],
        autoencoder_epochs=best_params['ae_epochs'],
        train_autoencoder_if_missing=True
    )
    
    model = CausalDilatedTCNModel(config=config)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    test_preds = model.predict(X_test)
    from sklearn.metrics import accuracy_score, classification_report
    test_accuracy = accuracy_score(y_test, (test_preds > 0.5).astype(int))
    
    logger.info(f"\n📊 Test Set Results:")
    logger.info(f"   Accuracy: {test_accuracy:.4f}")
    logger.info("\n" + classification_report(y_test, (test_preds > 0.5).astype(int)))


def example_2_advanced_hpo_with_custom_config():
    """Example 2: Advanced HPO with custom configuration."""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 2: Advanced HPO with Custom Configuration")
    logger.info("="*80 + "\n")
    
    # Generate data
    X_train, y_train, X_val, y_val, X_test, y_test = generate_synthetic_market_data(
        n_samples=2000,
        n_features=120,
        role="tactician"
    )
    
    # Create HPO instance with custom config
    from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import OptimizationStage
    
    hpo = AutoencoderTCNHPO(
        role="tactician",
        metric="f1",  # Optimize F1 score instead of accuracy
        n_rounds=3,  # 3 rounds for more refinement
        stages=[
            OptimizationStage.COARSE_GRID,
            OptimizationStage.FINE_GRID,
            OptimizationStage.TPE
        ],
        enable_final_refinement=True,
        final_refinement_trials=100,  # More trials for final refinement
        save_results=True,
        results_dir="artifacts/hpo/tactician_custom",
        verbose=True
    )
    
    # Run optimization
    result = hpo.optimize(X_train, y_train, X_val, y_val)
    
    logger.info(f"\n✅ Best F1 Score: {result.best_score:.4f}")
    logger.info(f"   Total trials: {result.total_trials}")
    logger.info(f"   Total time: {result.total_time:.1f}s")


def example_3_compare_with_and_without_hpo():
    """Example 3: Compare performance with and without HPO."""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 3: Compare HPO vs Default Parameters")
    logger.info("="*80 + "\n")
    
    # Generate data
    X_train, y_train, X_val, y_val, X_test, y_test = generate_synthetic_market_data(
        n_samples=2000,
        n_features=120,
        role="analyst"
    )
    
    # Test 1: Default parameters
    logger.info("🔬 Test 1: Training with DEFAULT parameters...")
    default_config = CausalTCNConfig(
        use_autoencoder=True,
        latent_dim=16,
        num_filters=64,
        num_layers=4,
        kernel_size=3,
        dilation_base=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        autoencoder_epochs=30
    )
    
    model_default = CausalDilatedTCNModel(config=default_config)
    model_default.fit(X_train, y_train)
    preds_default = model_default.predict(X_val)
    
    from sklearn.metrics import accuracy_score
    acc_default = accuracy_score(y_val, (preds_default > 0.5).astype(int))
    logger.info(f"   Default accuracy: {acc_default:.4f}\n")
    
    # Test 2: HPO-optimized parameters
    logger.info("🔬 Test 2: Training with HPO-OPTIMIZED parameters...")
    best_params, best_score = optimize_analyst_autoencoder_tcn(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        metric="accuracy",
        n_rounds=2,
        save_results=False
    )
    
    logger.info(f"   HPO accuracy: {best_score:.4f}\n")
    
    # Compare
    improvement = (best_score - acc_default) / acc_default * 100
    
    logger.info("\n" + "="*80)
    logger.info("📊 COMPARISON RESULTS")
    logger.info("="*80)
    logger.info(f"   Default parameters:     {acc_default:.4f}")
    logger.info(f"   HPO-optimized params:   {best_score:.4f}")
    logger.info(f"   Improvement:            {improvement:+.2f}%")
    logger.info("="*80)
    
    if improvement > 2:
        logger.info("✅ HPO provides significant improvement! (+2% or more)")
    elif improvement > 0:
        logger.info("✅ HPO provides modest improvement")
    else:
        logger.info("⚠️ Default parameters were already near-optimal")


def example_4_extract_insights_from_hpo():
    """Example 4: Extract insights from HPO results."""
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 4: Extract Insights from HPO Results")
    logger.info("="*80 + "\n")
    
    # Generate data
    X_train, y_train, X_val, y_val, _, _ = generate_synthetic_market_data(
        n_samples=2000,
        n_features=120,
        role="analyst"
    )
    
    # Run optimization
    hpo = AutoencoderTCNHPO(
        role="analyst",
        metric="accuracy",
        n_rounds=2,
        save_results=True
    )
    
    result = hpo.optimize(X_train, y_train, X_val, y_val)
    
    # Extract insights
    logger.info("\n📊 HPO INSIGHTS")
    logger.info("="*80)
    
    # Best parameters per group
    logger.info("\n1. Best Parameters by Group:")
    for group_name, group_result in result.group_results.items():
        logger.info(f"\n   {group_name}:")
        for param, value in group_result.best_params.items():
            logger.info(f"      {param}: {value}")
        logger.info(f"      → Score: {group_result.best_score:.4f}")
        logger.info(f"      → Trials: {group_result.n_trials}")
    
    # Parameter importance (which groups had biggest impact)
    logger.info("\n2. Parameter Group Impact:")
    logger.info("   (Score improvement when optimizing each group)")
    
    baseline_score = 0.5  # Random baseline
    for group_name, group_result in result.group_results.items():
        impact = group_result.best_score - baseline_score
        logger.info(f"   {group_name}: +{impact:.4f}")
        baseline_score = group_result.best_score
    
    # Optimization efficiency
    logger.info(f"\n3. Optimization Efficiency:")
    logger.info(f"   Total trials: {result.total_trials}")
    logger.info(f"   Total time: {result.total_time:.1f}s")
    logger.info(f"   Time per trial: {result.total_time/result.total_trials:.1f}s")
    logger.info(f"   Best score: {result.best_score:.4f}")
    
    # Key findings
    logger.info("\n4. Key Findings:")
    logger.info(f"   Optimal latent dimension: {result.best_params['latent_dim']}")
    logger.info(f"   Optimal TCN depth: {result.best_params['num_layers']} layers")
    logger.info(f"   Optimal TCN width: {result.best_params['num_filters']} filters")
    logger.info(f"   Compression ratio: {X_train.shape[1]}/{result.best_params['latent_dim']} = "
                f"{X_train.shape[1]/result.best_params['latent_dim']:.1f}x")


def main():
    """Run all examples."""
    logger.info("🚀 Autoencoder + TCN Hyperparameter Optimization Examples")
    logger.info("="*80)
    logger.info("These examples demonstrate hierarchical HPO for the enhanced TCN")
    logger.info("with autoencoder compression using tools from:")
    logger.info("   src/utils/ml_common/optimization/hierarchical_parameter_optimizer.py")
    logger.info("="*80)
    
    try:
        # Run examples
        example_1_basic_analyst_hpo()
        example_2_advanced_hpo_with_custom_config()
        example_3_compare_with_and_without_hpo()
        example_4_extract_insights_from_hpo()
        
        logger.info("\n" + "="*80)
        logger.info("✅ All HPO examples completed successfully!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

