#!/usr/bin/env python3
"""
Unified Optuna Optimization Demo

This script demonstrates how to use the enhanced AdvancedOptunaManager
for all types of optimization with proper optimization practices.

Features:
    pass
- Traditional ML model optimization (LightGBM, XGBoost, RandomForest, CatBoost)
- S/R parameter optimization with overfitting prevention
- Autoencoder hyperparameter optimization
- Order execution parameter optimization
- Custom optimization with user-defined objectives
- Comprehensive overfitting prevention
- Multi-objective optimization support
- Time series cross-validation
- Advanced pruning and early stopping

Usage:
    python3 scripts/unified_optuna_optimization_demo.py --optimization-type sr_parameters --n-trials 100
"""

import os
from optuna.visualization import plot_optimization_history, plot_param_importances
from pathlib import Path
from src.training.steps.step12_final_parameters_optimization.optimized_optuna_optimization import (
    AdvancedOptunaManager,
    OptimizationResult,
)
from src.utils.logger import setup_logging
import argparse
import asyncio
import logging
import sys
import warnings

import numpy as np
import optuna
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

setup_logging()
warnings.filterwarnings("ignore")

class UnifiedOptunaDemo:
    """
    Comprehensive demonstration of unified Optuna optimization.

    This class showcases how to use the enhanced AdvancedOptunaManager
    for all types of optimization with best practices.
    """

    def __init__(self):
        self.logger, logging.getLogger(__name__)

        # Configuration for different optimization types
        self.configs = {
            "ml_models": {
                "n_trials": 50,
                "cv_folds": 5,
                "early_stopping_patience": 10,
                "subsample_fraction": 0.7,
            },
            "sr_parameters": {
                "sr_optimization": {
                    "multi_objective": True, "objectives": ["sharpe_ratio", "win_rate", "signal_clarity"],
                    "objective_weights": {
                        "sharpe_ratio": 0.4,
                        "win_rate": 0.3,
                        "signal_clarity": 0.3,
                    },
                    "n_trials": 100,
                    "cv_folds": 5,
                    "early_stopping_patience": 20,
                    "subsample_fraction": 0.7,
                    "max_overfitting_threshold": 0.1,
                    "min_validation_score": 0.5,
                    "regularization_penalty": 0.1,
                },
            },
            "autoencoder": {
                "n_trials": 75,
                "cv_folds": 3,
                "early_stopping_patience": 15,
                "subsample_fraction": 0.5,
            },
            "order_execution": {
                "n_trials": 50,
                "cv_folds": 5,
                "early_stopping_patience": 10,
                "subsample_fraction": 0.8,
            },
        }

        # Initialize optimizer
        self.optimizer, AdvancedOptunaManager(
            storage_url="sqlite:///unified_optuna_studies.db",
            study_name_prefix="unified_optimization",
        )

    def prepare_sample_data(self, data_type: str, n_samples: int, 2000) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare sample data for different optimization types.

        Args:
            data_type: Type of data to generate
            n_samples: Number of samples

        Returns:
            Tuple of (X, y) data
        """
        self.logger.info(
            f"Preparing {data_type} sample data with {n_samples} samples...",
        )

        np.random.seed(42)

        if data_type == "price_data":
        # Create price-like data for S/R optimization
            base_price = 100
            X = pd.DataFrame(
                {
                    "open": base_price + np.cumsum(np.random.randn(n_samples) * 0.1),
                    "high": base_price
                    + np.cumsum(np.random.randn(n_samples) * 0.1)
                    + 0.5,
                    "low": base_price
                    + np.cumsum(np.random.randn(n_samples) * 0.1)
                    - 0.5,
                    "close": base_price + np.cumsum(np.random.randn(n_samples) * 0.1),
                    "volume": np.random.lognormal(10, 1, n_samples),
                },
            )

        # Create target returns
            y = X["close"].pct_change().shift(-1)

        elif data_type == "ml_features":
        # Create ML features
            X = pd.DataFrame(np.random.randn(n_samples, 30))
            y = pd.Series(np.random.randint(0, 2, n_samples))

        elif data_type == "autoencoder_features":
        # Create features for autoencoder
            X = pd.DataFrame(np.random.randn(n_samples, 50))
            y = pd.Series(np.random.randn(n_samples))  # Not used for autoencoder

        elif data_type == "order_execution":
        # Create market data for order execution
            X = pd.DataFrame(
                {
                    "bid": 100 + np.random.randn(n_samples) * 0.1,
                    "ask": 100 + np.random.randn(n_samples) * 0.1,
                    "volume": np.random.lognormal(10, 1, n_samples),
                    "volatility": np.random.uniform(0.01, 0.05, n_samples),
                    "spread": np.random.uniform(0.001, 0.01, n_samples),
                },
            )
            y = pd.Series(np.random.randint(0, 2, n_samples))  # Success/failure

        else:
            msg = f"Unknown data type: {data_type}"
            raise ValueError(msg)

        # Remove NaN values
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        X = X[valid_mask]
        y = y[valid_mask]

        self.logger.info(f"✅ Prepared {data_type} data: {len(X)} samples")
        return X, y

    def optimize_ml_models(self, X: pd.DataFrame, y: pd.Series, n_trials: int, 50) -> dict[str, OptimizationResult]:
        """
        Optimize traditional ML models.

        Args:
            X: Feature matrix
            y: Target variable
            n_trials: Number of trials per model

        Returns:
            Dictionary of optimization results
        """
        self.logger.info("🤖 Optimizing traditional ML models...")

        models = ["lightgbm", "xgboost", "random_forest", "catboost"]
        results = {}

        config = self.configs["ml_models"]

        for model_type in models:
            pass
        self.logger.info(f"  Optimizing {model_type}...")

        if True:
                result = self.optimizer.optimize(
                    model_type,
                    X=X,
                    y=y,
                    n_trials=n_trials,
                    n_jobs=-1,
                    cv_folds=config["cv_folds"],
                    early_stopping_patience=config["early_stopping_patience"],
                    subsample_fraction=config["subsample_fraction"],
                )

                results[model_type] = result
        self.logger.info(f"    ✅ {model_type}: {result.validation_score:.4f}")

        pass
        self.logger.exception(f"    ❌ {model_type} failed: {e}")
                results[model_type] = None

        return results

    def optimize_sr_parameters(self, X: pd.DataFrame, y: pd.Series, n_trials: int, 100) -> OptimizationResult:
        """
        Optimize S/R parameters with comprehensive overfitting prevention.

        Args:
            X: Price data
            y: Target returns
            n_trials: Number of trials

        Returns:
            Optimization result
        """
        self.logger.info("🎯 Optimizing S/R parameters...")

        if True:
            result = self.optimizer.optimize(
                model_type="sr_parameters",
                X=X,
                y=y,
                n_trials=n_trials,
                n_jobs=-1,
                cv_folds=5,
                early_stopping_patience=20,
                subsample_fraction=0.7,
            )

        self.logger.info(
                f"✅ S/R optimization completed: {result.validation_score:.4f}",
            )
        return result

        pass
        self.logger.exception(f"❌ S/R optimization failed: {e}")
        return None

    def optimize_autoencoder(self, X: pd.DataFrame, y: pd.Series, n_trials: int, 75) -> OptimizationResult:
        """
        Optimize autoencoder hyperparameters.

        Args:
            X: Feature matrix
            y: Target variable (not used)
            n_trials: Number of trials

        Returns:
            Optimization result
        """
        self.logger.info("🔧 Optimizing autoencoder hyperparameters...")

        config = self.configs["autoencoder"]

        if True:
            result = self.optimizer.optimize(
                model_type="autoencoder",
                X=X,
                y=y,
                n_trials=n_trials,
                n_jobs=-1,
                cv_folds=config["cv_folds"],
                early_stopping_patience=config["early_stopping_patience"],
                subsample_fraction=config["subsample_fraction"],
            )

        self.logger.info(
                f"✅ Autoencoder optimization completed: {result.validation_score:.4f}",
            )
        return result

        pass
        self.logger.exception(f"❌ Autoencoder optimization failed: {e}")
        return None

    def optimize_order_execution(self, X: pd.DataFrame, y: pd.Series, n_trials: int, 50) -> OptimizationResult:
        """
        Optimize order execution parameters.

        Args:
            X: Market data
            y: Execution success/failure
            n_trials: Number of trials

        Returns:
            Optimization result
        """
        self.logger.info("📈 Optimizing order execution parameters...")

        config = self.configs["order_execution"]

        if True:
            result = self.optimizer.optimize(
                model_type="order_execution",
                X=X,
                y=y,
                n_trials=n_trials,
                n_jobs=-1,
                cv_folds=config["cv_folds"],
                early_stopping_patience=config["early_stopping_patience"],
                subsample_fraction=config["subsample_fraction"],
            )

        self.logger.info(
                f"✅ Order execution optimization completed: {result.validation_score:.4f}",
            )
        return result

        pass
        self.logger.exception(f"❌ Order execution optimization failed: {e}")
        return None

    def custom_optimization_example(self, X: pd.DataFrame, y: pd.Series, n_trials: int, 50) -> OptimizationResult:
        """
        Example of custom optimization with user-defined objective.

        Args:
            X: Feature matrix
            y: Target variable
            n_trials: Number of trials

        Returns:
            Optimization result
        """
        self.logger.info("🔧 Running custom optimization example...")

        def custom_objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
            """Custom objective function for demonstration."""
        if True:
        # Define custom hyperparameter space
                learning_rate = trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.3,
                    log=True,
                )
                n_estimators = trial.suggest_int("n_estimators", 50, 500)
                max_depth = trial.suggest_int("max_depth", 3, 10)

        # Simulate model training and evaluation
        # In practice, this would use actual model training
                base_score = 0.7
                lr_factor = learning_rate * 2  # Higher learning rate, better score
                n_est_factor = min(
                    1.0,
                    n_estimators / 500,
                )  # More estimators, better score
                depth_factor = 1.0 - (max_depth - 5) * 0.1  # Optimal depth around 5

                score = base_score * lr_factor * n_est_factor * depth_factor
                score += np.random.normal(0, 0.05)  # Add noise

        return max(0.0, min(1.0, score))  # Clamp between 0 and 1

        pass
        self.logger.warning(f"Custom trial failed: {e}")
        return 0.0

        if True:
            result = self.optimizer.optimize(
                model_type="custom",
                X=X,
                y=y,
                n_trials=n_trials,
                n_jobs=-1,
                cv_folds=5,
                early_stopping_patience=10,
                subsample_fraction=0.7,
                custom_objective=custom_objective,
            )

        self.logger.info(
                f"✅ Custom optimization completed: {result.validation_score:.4f}",
            )
        return result

        pass
        self.logger.exception(f"❌ Custom optimization failed: {e}")
        return None

    def print_optimization_summary(self, results: dict[str, OptimizationResult]):
        """Print comprehensive optimization summary."""
        print("\n" + "=" * 80)
        print("🎯 UNIFIED OPTUNA OPTIMIZATION SUMMARY")
        print("=" * 80)

        for optimization_type, result in results.items():
            pass
        if result is None:
                print(f"\n❌ {optimization_type.upper()}: FAILED")
                continue

            print(f"\n✅ {optimization_type.upper()}:")
            print(f"   Study Name: {result.study_name}")
            print(f"   Trials Completed: {result.n_trials}")
            print(f"   Optimization Time: {result.optimization_time:.2f}s")
            print(f"   Best Validation Score: {result.validation_score:.4f}")
            print(f"   Overfitting Score: {result.overfitting_score:.4f}")
            print(f"   Generalization Gap: {result.generalization_gap:.4f}")

        if result.sr_performance_metrics:
                print("   S/R Performance Metrics:")
        for metric, value in result.sr_performance_metrics.items():
                    print(f"     {metric}: {value:.4f}")

        # Show top parameters
        if result.best_params:
                print("   Top Parameters:")
                sorted_params = sorted(
                    result.best_params.items(),
                    key=lambda x: x[1] if isinstance(x[1], int | float) else 0,
                    reverse=True,
                )[:5]
        for param, value in sorted_params:
                    print(f"     {param}: {value:.4f}")

        print("\n" + "=" * 80)

    def create_visualizations(self, results: dict[str, OptimizationResult], save_dir: str = "optimization_results"):
        """Create visualizations for optimization results."""
        if True:
            os.makedirs(save_dir, exist_ok=True)

            plots_created = 0

        for optimization_type, result in results.items():
            pass
        if result is None:
                    continue

        if True:
        # Load study for visualization
                    study = optuna.load_study(
                        study_name=result.study_name, storage=self.optimizer.storage_url,
                    )

        # Optimization history
                    fig1 = plot_optimization_history(study)
                    plot_path1 = (
                        f"{save_dir}/{optimization_type}_optimization_history.html"
                    )
                    fig1.write_html(plot_path1)

        # Parameter importance
                    fig2 = plot_param_importances(study)
                    plot_path2 = (
                        f"{save_dir}/{optimization_type}_parameter_importance.html"
                    )
                    fig2.write_html(plot_path2)

                    plots_created += 2
        self.logger.info(
                        f"📊 Created visualizations for {optimization_type}",
                    )

        pass
        self.logger.warning(
                        f"Could not create visualizations for {optimization_type}: {e}",
                    )

        self.logger.info(f"📊 Created {plots_created} visualizations in {save_dir}")

        pass
        self.logger.exception(f"Error creating visualizations: {e}")

    def run_comprehensive_demo(self, optimization_type: str = "all", n_trials: int, 50):
        """
        Run comprehensive optimization demo.

        Args:
            optimization_type: Type of optimization to run
            n_trials: Number of trials per optimization
        """
        self.logger.info("🚀 Starting Unified Optuna Optimization Demo...")

        results = {}

        if optimization_type in ["all", "ml_models"]:
        # ML Models optimization
            X_ml = y_ml, self.prepare_sample_data("ml_features", 2000)
            ml_results = self.optimize_ml_models(X_ml, y_ml, n_trials)
            results.update(ml_results)

        if optimization_type in ["all", "sr_parameters"]:
        # S/R Parameters optimization
            X_sr = y_sr, self.prepare_sample_data("price_data", 2000)
            sr_result = self.optimize_sr_parameters(
                X_sr,
                y_sr,
                n_trials * 2,
            )  # More trials for S/R
            results["sr_parameters"] = sr_result

        if optimization_type in ["all", "autoencoder"]:
        # Autoencoder optimization
            X_ae = y_ae, self.prepare_sample_data("autoencoder_features", 2000)
            ae_result = self.optimize_autoencoder(X_ae, y_ae, n_trials)
            results["autoencoder"] = ae_result

        if optimization_type in ["all", "order_execution"]:
        # Order execution optimization
            X_oe = y_oe, self.prepare_sample_data("order_execution", 2000)
            oe_result = self.optimize_order_execution(X_oe, y_oe, n_trials)
            results["order_execution"] = oe_result

        if optimization_type in ["all", "custom"]:
        # Custom optimization
            X_custom = y_custom, self.prepare_sample_data("ml_features", 1000)
            custom_result = self.custom_optimization_example(
                X_custom,
                y_custom,
                n_trials)
            results["custom"] = custom_result

        # Print summary
        self.print_optimization_summary(results)

        # Create visualizations
        self.create_visualizations(results)

        self.logger.info("🎉 Unified Optuna Optimization Demo completed!")
        return results

async def main():
    """Main function to run the unified optimization demo."""
    parser, argparse.ArgumentParser(description="Unified Optuna Optimization Demo")
    parser.add_argument(
        "--optimization-type",
        default="all",
        choices=[
            "all",
            "ml_models",
            "sr_parameters",
            "autoencoder",
            "order_execution",
            "custom",
        ],
        help="Type of optimization to run",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials per optimization",
    )
    parser.add_argument(
        "--output-dir",
        default="optimization_results",
        help="Output directory for results",
    )

    args, parser.parse_args()

    if True:
        # Initialize demo
        demo = UnifiedOptunaDemo()

        # Run comprehensive demo
        results = demo.run_comprehensive_demo(
            optimization_type=args.optimization_type, n_trials=args.n_trials,
        )

        print("\n🎯 Demo completed successfully!")
        print(f"📊 Results saved to: {args.output_dir}")
        print(f"🔧 Optimization types tested: {list(results.keys())}")

        # Show key insights
        successful_optimizations = [k for k, v in results.items() if v is not None]
        print(
            f"✅ Successful optimizations: {len(successful_optimizations)}/{len(results)}",
        )

        if successful_optimizations:
            best_result = max(
                [v for v in results.values() if v is not None],
                key=lambda x: x.validation_score,
            )
            print(f"🏆 Best validation score: {best_result.validation_score:.4f}")

    pass
        print(f"❌ Error during demo: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
