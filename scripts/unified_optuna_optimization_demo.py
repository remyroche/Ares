#!/usr/bin/env python3
"""
Unified Optuna Optimization Demo

This script demonstrates how to use the enhanced AdvancedOptunaManager
for different types of optimization with proper practices.

Features:
- Traditional ML model optimization (LightGBM, XGBoost, RandomForest, CatBoost)
- S/R-like parameter optimization (demonstrated via a surrogate model)
- Autoencoder-like optimization (demonstrated via a surrogate model)
- Order execution parameter optimization (surrogate model)
- Custom optimization with user-defined objectives
- Multi-objective ideas outlined (single objective in demo)
- Time series cross-validation (as configured in manager)
- Pruning and early stopping as provided by manager

Usage:
    python3 scripts/unified_optuna_optimization_demo.py --optimization-type sr_parameters --n-trials 100
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
import pandas as pd
from optuna.visualization import plot_optimization_history, plot_param_importances

# Ensure src on path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.training.steps.step17_final_parameters_optimization.optimized_optuna_optimization import (  # noqa: E402
    AdvancedOptunaManager,
)
from src.utils.logger import setup_logging  # noqa: E402


# Initialize logging and suppress noisy warnings
setup_logging()
warnings.filterwarnings("ignore")


class UnifiedOptunaDemo:
    """
    Comprehensive demonstration of unified Optuna optimization.

    This class showcases how to use the AdvancedOptunaManager
    for multiple optimization scenarios with best practices.
    """

    def __init__(self) -> None:
        self.logger=logging.getLogger(__name__)

        # Configuration for different optimization types
        self.configs: dict[str, dict[str, Any]] = {
            "ml_models": {
                "n_trials": 50,
                "cv_folds": 5,
                "early_stopping_patience": 10,
                "subsample_fraction": 0.7,
            },
            "sr_parameters": {
                "sr_optimization": {
                    "multi_objective": True,
                    "objectives": ["sharpe_ratio", "win_rate", "signal_clarity"],
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
        self.optimizer=AdvancedOptunaManager(
            storage_url="sqlite:///unified_optuna_studies.db",
            study_name_prefix="unified_optimization",
        )

    def prepare_sample_data(self, data_type: str, n_samples: int=2000) -> tuple[pd.DataFrame, pd.Series]:
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

        rng=np.random.default_rng(42)

        if data_type== "price_data":
            # Create price-like data for S/R optimization
            base_price = 100.0
            X = pd.DataFrame(
                {
                    "open": base_price + np.cumsum(rng.normal(0, 0.1, n_samples)),
                    "high": base_price + np.cumsum(rng.normal(0, 0.1, n_samples)) + 0.5,
                    "low": base_price + np.cumsum(rng.normal(0, 0.1, n_samples)) - 0.5,
                    "close": base_price + np.cumsum(rng.normal(0, 0.1, n_samples)),
                    "volume": rng.lognormal(10, 1, n_samples),
                }
            )
            # Create target returns
            y=X["close"].pct_change().shift(-1).fillna(0.0)
            y=(y > 0).astype(int)  # Convert to a classification target

        elif data_type== "ml_features":
            # Create ML features
            X = pd.DataFrame(rng.normal(size=(n_samples, 30)))
            y=pd.Series(rng.integers(0, 2, size=n_samples))

        elif data_type== "autoencoder_features":
            # Create features for autoencoder (use surrogate classification target)
            X=pd.DataFrame(rng.normal(size=(n_samples, 50)))
            y=pd.Series((X.mean(axis=1) > 0).astype(int))

        elif data_type== "order_execution":
            # Create market data for order execution
            X = pd.DataFrame(
                {
                    "bid": 100 + rng.normal(0, 0.1, n_samples),
                    "ask": 100 + rng.normal(0, 0.1, n_samples),
                    "volume": rng.lognormal(10, 1, n_samples),
                    "volatility": rng.uniform(0.01, 0.05, n_samples),
                    "spread": rng.uniform(0.001, 0.01, n_samples),
                }
            )
            y=pd.Series(rng.integers(0, 2, size=n_samples))  # Success/failure

        else:
            msg=f"Unknown data type: {data_type}"
            raise ValueError(msg)

        # Remove NaN values
        valid_mask=~(y.isna() | X.isna().any(axis=1))
        X=X[valid_mask].reset_index(drop=True)
        y=y[valid_mask].reset_index(drop=True)

        self.logger.info(f"✅ Prepared {data_type} data: {len(X)} samples")
        return X, y

    def optimize_ml_models(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int=50
    ) -> dict[str, Optional[dict[str, Any]]]:
        """
        Optimize traditional ML models.

        Args:
            X: Feature matrix
            y: Target variable
            n_trials: Number of trials per model

        Returns:
            Dictionary of optimization results (per model)
        """
        self.logger.info("🤖 Optimizing traditional ML models...")

        models=["lightgbm", "xgboost", "random_forest", "catboost"]
        results: dict[str, Optional[dict[str, Any]]] = {}

        config=self.configs["ml_models"]

        for model_type in models:
            try:
                self.logger.info(f"  Optimizing {model_type}...")
                result=self.optimizer.optimize(
                    model_type,
                    X=X,
                    y=y,
                    n_trials=n_trials,
                    n_jobs=-1,
                    cv_folds=int(config["cv_folds"]),
                    early_stopping_patience=int(config["early_stopping_patience"]),
                    subsample_fraction=float(config["subsample_fraction"]),
                )
                results[model_type] = result
                self.logger.info(f"    ✅ {model_type}: best_value={result.get('best_value', float('nan')):.4f}")
            except Exception as e:
                self.logger.exception(f"    ❌ {model_type} failed: {e}")
                results[model_type] = None

        return results

    def optimize_sr_parameters(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int=100
    ) -> Optional[dict[str, Any]]:
        """
        Optimize S/R-like parameters using a surrogate model configuration.
        Demonstrated via XGBoost to keep the demo self-contained.
        """
        self.logger.info("🎯 Optimizing S/R parameters (surrogate via XGBoost)...")

        try:
            result=self.optimizer.optimize(
                model_type="xgboost",
                X=X,
                y=y,
                n_trials=n_trials,
                n_jobs=-1,
                cv_folds=5,
                early_stopping_patience=20,
                subsample_fraction=0.7,
            )
            self.logger.info(
                f"✅ S/R optimization completed: best_value={result.get('best_value', float('nan')):.4f}",
            )
            return result
        except Exception as e:
            self.logger.exception(f"❌ S/R optimization failed: {e}")
            return None

    def optimize_autoencoder(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int=75
    ) -> Optional[dict[str, Any]]:
        """
        Optimize autoencoder-like hyperparameters using a surrogate model (LightGBM).
        """
        self.logger.info("🔧 Optimizing autoencoder hyperparameters (surrogate via LightGBM)...")

        config=self.configs["autoencoder"]

        try:
            result = self.optimizer.optimize(
                model_type="lightgbm",
                X=X,
                y=y,
                n_trials=n_trials,
                n_jobs=-1,
                cv_folds=int(config["cv_folds"]),
                early_stopping_patience=int(config["early_stopping_patience"]),
                subsample_fraction=float(config["subsample_fraction"]),
            )
            self.logger.info(
                f"✅ Autoencoder optimization completed: best_value={result.get('best_value', float('nan')):.4f}",
            )
            return result
        except Exception as e:
            self.logger.exception(f"❌ Autoencoder optimization failed: {e}")
            return None

    def optimize_order_execution(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int=50
    ) -> Optional[dict[str, Any]]:
        """
        Optimize order execution parameters using a surrogate model (RandomForest).
        """
        self.logger.info("📈 Optimizing order execution parameters (surrogate via RandomForest)...")

        config=self.configs["order_execution"]

        try:
            result = self.optimizer.optimize(
                model_type="random_forest",
                X=X,
                y=y,
                n_trials=n_trials,
                n_jobs=-1,
                cv_folds=int(config["cv_folds"]),
                early_stopping_patience=int(config["early_stopping_patience"]),
                subsample_fraction=float(config["subsample_fraction"]),
            )
            self.logger.info(
                f"✅ Order execution optimization completed: best_value={result.get('best_value', float('nan')):.4f}",
            )
            return result
        except Exception as e:
            self.logger.exception(f"❌ Order execution optimization failed: {e}")
            return None

    def custom_optimization_example(
        self, X: pd.DataFrame, y: pd.Series, n_trials: int=50
    ) -> Optional[dict[str, Any]]:
        """
        Example of custom optimization with user-defined objective.
        Implemented by wrapping a score into the XGBoost objective space.
        """
        self.logger.info("🔧 Running custom optimization example...")

        def custom_objective(trial: optuna.Trial, X_df: pd.DataFrame, y_sr: pd.Series) -> float:
            """Custom objective function for demonstration."""
            # Define custom hyperparameter space
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            n_estimators=trial.suggest_int("n_estimators", 50, 500)
            max_depth=trial.suggest_int("max_depth", 3, 10)

            # Simulate model training and evaluation (proxy score)
            base_score=0.7
            lr_factor = learning_rate * 2.0  # Higher learning rate, better score
            n_est_factor=min(1.0, n_estimators / 500.0)  # More estimators, better score
            depth_factor=1.0 - (max_depth - 5) * 0.1  # Optimal depth around 5

            score=base_score * lr_factor * n_est_factor * depth_factor
            score += float(np.random.normal(0, 0.05))  # Add noise
            # Clamp between 0 and 1
            return float(max(0.0, min(1.0, score)))

        try:
            # Use manager's study infra with the custom objective by mapping to a supported model
            # We directly run an Optuna study here to showcase custom objective
            study_name=f"{self.optimizer.study_name_prefix}_custom"
            study = optuna.create_study(
                storage=self.optimizer.storage_url,
                study_name=study_name,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=n_trials),
                load_if_exists=True,
            )

            def _objective(trial: optuna.Trial) -> float:
                return custom_objective(trial, X, y)

            study.optimize(_objective, n_trials=n_trials, n_jobs=1)

            result: dict[str, Any] = {
                "study_name": study.study_name,
                "best_value": float(study.best_value),
                "best_params": dict(study.best_params),
                "total_trials": int(len(study.trials)),
                "n_completed": int(
                    len(
                        study.get_trials(
                            deepcopy=False,
                            states=[optuna.trial.TrialState.COMPLETE],
                        )
                    )
                ),
                "n_pruned": int(
                    len(
                        study.get_trials(
                            deepcopy=False,
                            states=[optuna.trial.TrialState.PRUNED],
                        )
                    )
                ),
            }
            self.logger.info(
                f"✅ Custom optimization completed: best_value={result.get('best_value', float('nan')):.4f}",
            )
            return result
        except Exception as e:
            self.logger.exception(f"❌ Custom optimization failed: {e}")
            return None

    def print_optimization_summary(self, results: dict[str, Optional[dict[str, Any]]]) -> None:
        """Print comprehensive optimization summary."""
        print("\n" + "=" * 80)
        print("🎯 UNIFIED OPTUNA OPTIMIZATION SUMMARY")
        print("=" * 80)

        for optimization_type, result in results.items():
            if result is None:
                print(f"\n❌ {optimization_type.upper()}: FAILED")
                continue

            print(f"\n✅ {optimization_type.upper()}:")
            print(f"   Study Name: {result.get('study_name', 'N/A')}")
            print(f"   Trials Completed: {result.get('total_trials', 'N/A')}")
            print(f"   Best Value: {result.get('best_value', float('nan')):.4f}")

            # Show top parameters
            best_params=result.get("best_params", {})
            if best_params:
                print("   Top Parameters:")
                # Sort numeric params by value for display
                try:
                    sorted_params=sorted(
                        best_params.items(),
                        key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0.0,
                        reverse=True,
                    )[:5]
                except Exception:
                    sorted_params=list(best_params.items())[:5]
                for param, value in sorted_params:
                    print(f"     {param}: {value}")

        print("\n" + "=" * 80)

    def create_visualizations(
        self, results: dict[str, Optional[dict[str, Any]]], save_dir: str="optimization_results"
    ) -> None:
        """Create visualizations for optimization results."""
        try:
            os.makedirs(save_dir, exist_ok=True)
            plots_created=0

            for optimization_type, result in results.items():
                if result is None:
                    continue

                try:
                    # Load study for visualization
                    study=optuna.load_study(
                        study_name=str(result["study_name"]), storage=self.optimizer.storage_url
                    )

                    # Optimization history
                    fig1=plot_optimization_history(study)
                    plot_path1=f"{save_dir}/{optimization_type}_optimization_history.html"
                    fig1.write_html(plot_path1)

                    # Parameter importance
                    fig2=plot_param_importances(study)
                    plot_path2=f"{save_dir}/{optimization_type}_parameter_importance.html"
                    fig2.write_html(plot_path2)

                    plots_created += 2
                    self.logger.info(
                        f"📊 Created visualizations for {optimization_type}",
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Could not create visualizations for {optimization_type}: {e}",
                    )

            self.logger.info(f"📊 Created {plots_created} visualizations in {save_dir}")
        except Exception as e:
            self.logger.exception(f"Error creating visualizations: {e}")

    def run_comprehensive_demo(self, optimization_type: str="all", n_trials: int=50) -> dict[str, Optional[dict[str, Any]]]:
        """
        Run comprehensive optimization demo.

        Args:
            optimization_type: Type of optimization to run
            n_trials: Number of trials per optimization
        """
        self.logger.info("🚀 Starting Unified Optuna Optimization Demo...")

        results: dict[str, Optional[dict[str, Any]]] = {}

        if optimization_type in ["all", "ml_models"]:
            # ML Models optimization
            X_ml, y_ml=self.prepare_sample_data("ml_features", 2000)
            ml_results=self.optimize_ml_models(X_ml, y_ml, n_trials)
            results.update({f"ml_{k}": v for k, v in ml_results.items()})

        if optimization_type in ["all", "sr_parameters"]:
            # S/R Parameters optimization (surrogate)
            X_sr, y_sr=self.prepare_sample_data("price_data", 2000)
            sr_result=self.optimize_sr_parameters(
                X_sr,
                y_sr,
                n_trials * 2,
            )  # More trials for S/R
            results["sr_parameters"] = sr_result

        if optimization_type in ["all", "autoencoder"]:
            # Autoencoder optimization (surrogate)
            X_ae, y_ae=self.prepare_sample_data("autoencoder_features", 2000)
            ae_result=self.optimize_autoencoder(X_ae, y_ae, n_trials)
            results["autoencoder"] = ae_result

        if optimization_type in ["all", "order_execution"]:
            # Order execution optimization (surrogate)
            X_oe, y_oe=self.prepare_sample_data("order_execution", 2000)
            oe_result=self.optimize_order_execution(X_oe, y_oe, n_trials)
            results["order_execution"] = oe_result

        if optimization_type in ["all", "custom"]:
            # Custom optimization
            X_custom, y_custom=self.prepare_sample_data("ml_features", 1000)
            custom_result=self.custom_optimization_example(
                X_custom,
                y_custom,
                n_trials,
            )
            results["custom"] = custom_result

        # Print summary
        self.print_optimization_summary(results)

        # Create visualizations
        self.create_visualizations(results)

        self.logger.info("🎉 Unified Optuna Optimization Demo completed!")
        return results


async def main() -> int:
    """Main function to run the unified optimization demo."""
    parser=argparse.ArgumentParser(description="Unified Optuna Optimization Demo")
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

    args=parser.parse_args()

    try:
        # Initialize demo
        demo=UnifiedOptunaDemo()

        # Run comprehensive demo
        results=demo.run_comprehensive_demo(
            optimization_type=args.optimization_type, n_trials=args.n_trials,
        )

        print("\n🎯 Demo completed successfully!")
        print(f"📊 Results saved to: {args.output_dir}")
        print(f"🔧 Optimization types tested: {list(results.keys())}")

        # Show key insights
        successful_optimizations=[k for k, v in results.items() if v is not None]
        print(
            f"✅ Successful optimizations: {len(successful_optimizations)}/{len(results)}",
        )

        if successful_optimizations:
            # Use best_value field when available
            non_null_results=[v for v in results.values() if v is not None]
            best_result=max(
                non_null_results, key=lambda x: float(x.get("best_value", float("nan")))
            )
            print(f"🏆 Best value: {best_result.get('best_value', float('nan')):.4f}")
        return 0
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return 1


if __name__== "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
