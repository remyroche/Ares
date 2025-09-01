#!/usr/bin/env python3
"""
S/R Parameter Optimization Script

This script demonstrates how to use the enhanced HPO system to optimize
Support/Resistance parameters with comprehensive overfitting prevention.

Features:
- S/R Strength Score Weight Optimization
- Level Detection Parameter Tuning
- Breakout Threshold Optimization
- Zone Multiplier Tuning
- Confidence Threshold Optimization
- Overfitting Prevention with Time Series Cross-Validation
- Multi-Objective Optimization
- Comprehensive Performance Metrics

Usage:
    python3 scripts/run_sr_optimization.py --symbol ETHUSDT \\\
        --exchange BINANCE --period 365
"""

# ruff: noqa: E501, I001, C901, PLR2004


import argparse
import asyncio
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna.visualization import (
import plot_optimization_history,
    plot_optimization_history,
    plot_param_importances,
)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config_optuna import (  # noqa: E402
import SROptimizationParameters,
    SROptimizationParameters,
    validate_sr_optimization_config,
)
from src.training.steps.step17_final_parameters_optimization.optimized_optuna_optimization import (  # noqa: E402
import AdvancedOptunaManager,
    AdvancedOptunaManager,
    OptimizationResult,
)
from src.utils.logger import setup_logging  # noqa: E402

import setup_logging
setup_logging()
warnings.filterwarnings("ignore")


class SROptimizationRunner:
    """
    Comprehensive S/R parameter optimization runner with overfitting prevention.

    This class provides a complete workflow for optimizing S/R parameters:
    1. Data preparation and validation
    2. Configuration setup and validation
    3. Optimization execution with overfitting prevention
    4. Results analysis and visualization
    5. Parameter export and integration
    """

    def __init__(self, config: dict[str, Any]):
    pass
    pass
        self.config = config
        self.logger = logging.getLogger(__name__)

        # S/R optimization configuration
        self.sr_config = SROptimizationParameters()
        if "sr_optimization" in config:
    pass
    pass
            sr_config_dict = config["sr_optimization"]
            for key, value in sr_config_dict.items():
    pass
    pass
                if hasattr(self.sr_config, key):
    pass
    pass
                    setattr(self.sr_config, key, value)

        # Validate configuration
        if not validate_sr_optimization_config(self.sr_config):
    pass
    pass
            msg = "Invalid S/R optimization configuration"
            raise ValueError(msg)

        # Initialize optimizer
        self.optimizer = AdvancedOptunaManager(
            storage_url="sqlite:///sr_optuna_studies.db",
            study_name_prefix="sr_optimization",
            config=config,
        )

        # Results storage
        self.optimization_results: OptimizationResult | None = None
        self.study: optuna.Study | None = None

    def prepare_sample_data(self, n_samples: int = 2000) -> tuple[pd.DataFrame, pd.Series]:
    pass
    pass
        """
        Prepare sample price data for S/R optimization.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Tuple of (price_data, target_returns)
        """
        self.logger.info(f"Preparing sample data with {n_samples} samples...")

        rng = np.random.default_rng(42)

        # Create realistic price data
        base_price = 100
        price_data = pd.DataFrame(
            {
                "open": base_price + np.cumsum(rng.standard_normal(n_samples) * 0.1),
                "high": base_price
                + np.cumsum(rng.standard_normal(n_samples) * 0.1)
                + 0.5,
                "low": base_price
                + np.cumsum(rng.standard_normal(n_samples) * 0.1)
                - 0.5,
                "close": base_price + np.cumsum(rng.standard_normal(n_samples) * 0.1),
                "volume": rng.lognormal(10, 1, n_samples),
            },
        )

        # Calculate target returns (next period returns)
        target_returns = price_data["close"].pct_change().shift(-1)

        # Remove NaN values
        valid_mask = ~(target_returns.isna() | price_data.isna().any(axis=1))
        price_data = price_data[valid_mask]
        target_returns = target_returns[valid_mask]

        self.logger.info(f"✅ Prepared data: {len(price_data)} samples")
        self.logger.info(
            "   Price range: %.2f - %.2f",
            float(price_data["close"].min()),
            float(price_data["close"].max()),
        )
        self.logger.info(
            "   Returns range: %.4f - %.4f",
            float(target_returns.min()),
            float(target_returns.max()),
        )

        return price_data, target_returns

    async def run_optimization(
        self,
        price_data: pd.DataFrame,
        target_returns: pd.Series,
        n_trials: int = 100,
        study_name: str | None = None,
    ) -> OptimizationResult:
        """
        Run S/R parameter optimization with comprehensive overfitting prevention.

        Args:
            price_data: OHLCV price data
            target_returns: Target returns for optimization
            n_trials: Number of optimization trials
            study_name: Optional study name

        Returns:
            OptimizationResult with comprehensive metrics
        """
        self.logger.info("🚀 Starting S/R parameter optimization...")

        # Run optimization
        result = self.optimizer.optimize(
            model_type="sr_parameters",
            X=price_data,
            y=target_returns,
            n_trials=n_trials,
            n_jobs=-1,
            subsample_fraction=self.sr_config.subsample_fraction,
        )

        self.optimization_results = result

        # Load study for analysis
        resolved_study_name = study_name or "sr_optimization_sr_parameters"
        try:
            self.study = optuna.load_study(
                study_name=resolved_study_name,
                storage=self.optimizer.storage_url,
    except Exception as e:
        pass
    except Exception as e:
        pass
            )
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Could not load study for analysis: %s", e)

        self.logger.info("✅ S/R optimization completed successfully")
        return result

    def analyze_results(self) -> dict[str, Any]:
    pass
    pass
        """
        Analyze optimization results and generate comprehensive report.

        Returns:
            Dictionary with analysis results
        """
        if not self.optimization_results:
    pass
    pass
            msg = "No optimization results to analyze"
            raise ValueError(msg)

        self.logger.info("📊 Analyzing optimization results...")

        result = self.optimization_results

        # Basic metrics
        analysis: dict[str, Any] = {
            "optimization_summary": {
                "study_name": result.study_name,
                "trials_completed": result.n_trials,
                "optimization_time": result.optimization_time,
                "best_validation_score": result.validation_score,
                "overfitting_score": result.overfitting_score,
                "generalization_gap": result.generalization_gap,
            },
            "performance_metrics": result.sr_performance_metrics or {},
            "best_parameters": result.best_params,
            "overfitting_assessment": self._assess_overfitting(result),
            "parameter_importance": self._analyze_parameter_importance(),
            "recommendations": self._generate_recommendations(result),
        }

        self.logger.info("✅ Analysis completed")
        return analysis

    def _assess_overfitting(self, result: OptimizationResult) -> dict[str, Any]:
    pass
    pass
        """Assess overfitting based on optimization results."""
        overfit_threshold = 0.1
        low_severity_threshold = 0.05

        overfitting_assessment = {
            "is_overfitting": result.overfitting_score > overfit_threshold,
            "overfitting_severity": "low"
            if result.overfitting_score < low_severity_threshold
            else "medium"
            if result.overfitting_score < overfit_threshold
            else "high",
            "generalization_quality": "good"
            if result.generalization_gap < low_severity_threshold
            else "acceptable"
            if result.generalization_gap < overfit_threshold
            else "poor",
            "recommendations": [],
        }

        # Generate recommendations
        if overfitting_assessment["is_overfitting"]:
    pass
    pass
            overfitting_assessment["recommendations"].append(
                "Consider increasing regularization or reducing model complexity",
            )

        if result.generalization_gap > overfit_threshold:
    pass
    pass
            overfitting_assessment["recommendations"].append(
                "Validation set may not be representative of test set",
            )

        if result.validation_score < 0.5:
    pass
    pass
            overfitting_assessment["recommendations"].append(
                "Consider expanding the parameter search space",
            )

        return overfitting_assessment

    def _analyze_parameter_importance(self) -> dict[str, float]:
    pass
    pass
        """Analyze parameter importance from optimization study."""
        if not self.study:
    pass
    pass
            return {}

        try:
            # Get parameter importance
    except Exception as e:
        pass
    except Exception as e:
        pass
            return optuna.importance.get_param_importances(self.study)
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Could not calculate parameter importance: %s", e)
            return {}

    def _generate_recommendations(self, result: OptimizationResult) -> list[str]:
    pass
    pass
        """Generate actionable recommendations based on results."""
        recommendations: list[str] = []

        # Performance-based recommendations
        if result.validation_score < 0.6:
    pass
    pass
            recommendations.append(
                "Consider increasing the number of optimization trials",
            )
            recommendations.append("Review the parameter search space bounds")

        # Overfitting-based recommendations
        if result.overfitting_score > 0.1:
    pass
    pass
            recommendations.append(
                "Increase regularization in strength score calculation",
            )
            recommendations.append(
                "Reduce model complexity by limiting parameter ranges",
            )

        # Parameter-specific recommendations
        best_params = result.best_params

        # Check if weights are balanced
        weight_params = [k for k in best_params if "weight" in k]
        if weight_params:
    pass
    pass
            weights = [best_params[k] for k in weight_params]
            weight_std = float(np.std(weights))
            if weight_std > 0.2:
    pass
    pass
                recommendations.append(
                    "Consider rebalancing strength score weights for more stable "
                    "performance",
                )

        # Check confidence thresholds
        if best_params.get("min_sr_confidence", 0) < 0.6:
    pass
    pass
            recommendations.append(
                "Consider increasing minimum confidence threshold for better "
                "signal quality",
            )

        if best_params.get("high_confidence_threshold", 0) > 0.85:
    pass
    pass
            recommendations.append(
                "High confidence threshold may be too restrictive, consider "
                "lowering",
            )

        return recommendations

    def create_visualizations(self, save_dir: str = "optimization_results") -> dict[str, str]:
    pass
    pass
        """
        Create optimization visualizations.

        Args:
            save_dir: Directory to save visualizations

        Returns:
            Dictionary mapping plot names to file paths
        """
        if not self.study:
    pass
    pass
            self.logger.warning("No study available for visualization")
            return {}

        try:
            save_dir_path = Path(save_dir)
    except Exception as e:
        pass
    except Exception as e:
        pass
            save_dir_path.mkdir(parents=True, exist_ok=True)

            plots: dict[str, str] = {}

            # Optimization history
            fig1 = plot_optimization_history(self.study)
            plot_path1 = f"{save_dir}/optimization_history.html"
            fig1.write_html(plot_path1)
            plots["optimization_history"] = plot_path1

            # Parameter importance
            fig2 = plot_param_importances(self.study)
            plot_path2 = f"{save_dir}/parameter_importance.html"
            fig2.write_html(plot_path2)
            plots["parameter_importance"] = plot_path2

            self.logger.info("📊 Created %d visualizations in %s", len(plots), save_dir)
        except Exception:  # noqa: BLE001
            self.logger.exception("Error creating visualizations")
            return {}
        else:
            return plots

    def export_parameters(self, output_path: str = "optimized_sr_parameters.json") -> str:
    pass
    pass
        """
        Export optimized parameters to JSON file.

        Args:
            output_path: Path to save parameters

        Returns:
            Path to saved file
        """
        if not self.optimization_results:
    pass
    pass
            msg = "No optimization results to export"
            raise ValueError(msg)

        # Prepare parameters for export
        export_data: dict[str, Any] = {
            "optimization_metadata": {
                "study_name": self.optimization_results.study_name,
                "optimization_time": self.optimization_results.optimization_time,
                "n_trials": self.optimization_results.n_trials,
                "validation_score": self.optimization_results.validation_score,
                "overfitting_score": self.optimization_results.overfitting_score,
            },
            "strength_score_weights": {
                k: v
                for k, v in self.optimization_results.best_params.items()
                if "weight" in k
            },
            "level_detection_params": {
                k: v
                for k, v in self.optimization_results.best_params.items()
                if k
                in [
                    "min_touch_count",
                    "min_level_age_hours",
                    "price_tolerance_pct",
                    "volume_threshold",
                    "strength_threshold",
                ]
            },
            "breakout_thresholds": {
                k: v
                for k, v in self.optimization_results.best_params.items()
                if k
                in [
                    "breakout_threshold",
                    "confirmation_periods",
                    "volume_confirmation",
                    "momentum_threshold",
                    "false_breakout_filter",
                ]
            },
            "zone_multipliers": {
                k: v
                for k, v in self.optimization_results.best_params.items()
                if k
                in [
                    "support_zone_multiplier",
                    "resistance_zone_multiplier",
                    "sr_zone_threshold",
                    "zone_expansion_factor",
                    "zone_contraction_factor",
                ]
            },
            "confidence_thresholds": {
                k: v
                for k, v in self.optimization_results.best_params.items()
                if k
                in [
                    "min_sr_confidence",
                    "high_confidence_threshold",
                    "confidence_decay_rate",
                    "regime_confidence_boost",
                    "ensemble_confidence_threshold",
                ]
            },
        }

        # Save to file
        output_path_obj = Path(output_path)
        with output_path_obj.open("w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        self.logger.info("✅ Parameters exported to %s", output_path)
        return output_path

    def print_comprehensive_report(self, analysis: dict[str, Any]) -> None:
    pass
    pass
        """Print comprehensive optimization report."""
        print("\\\n" + "=" * 80)
        print("🎯 S/R PARAMETER OPTIMIZATION COMPREHENSIVE REPORT")
        print("=" * 80)

        # Optimization Summary
        summary = analysis["optimization_summary"]
        print("\\\n📊 OPTIMIZATION SUMMARY:")
        print(f"   Study Name: {summary['study_name']}")
        print(f"   Trials Completed: {summary['trials_completed']}")
        print(f"   Optimization Time: {summary['optimization_time']:.2f}s")
        print(f"   Best Validation Score: {summary['best_validation_score']:.4f}")
        print(f"   Overfitting Score: {summary['overfitting_score']:.4f}")
        print(f"   Generalization Gap: {summary['generalization_gap']:.4f}")

        # Performance Metrics
        if analysis["performance_metrics"]:
    pass
    pass
            print("\\\n📈 PERFORMANCE METRICS:")
            for metric, value in analysis["performance_metrics"].items():
    pass
    pass
                print(f"   {metric}: {value:.4f}")

        # Overfitting Assessment
        overfitting = analysis["overfitting_assessment"]
        print("\\\n🔍 OVERFITTING ASSESSMENT:")
        print(f"   Is Overfitting: {'Yes' if overfitting['is_overfitting'] else 'No'}")
        print(f"   Overfitting Severity: {overfitting['overfitting_severity'].title()}")
        print(
            f"   Generalization Quality: {overfitting['generalization_quality'].title()}",
        )

        if overfitting["recommendations"]:
    pass
    pass
            print("   Recommendations:")
            for rec in overfitting["recommendations"]:
    pass
    pass
                print(f"     • {rec}")

        # Parameter Importance
        if analysis["parameter_importance"]:
    pass
    pass
            print("\\\n⚙️ PARAMETER IMPORTANCE (Top 10):")
            sorted_importance = sorted(
                analysis["parameter_importance"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            for param, importance in sorted_importance:
    pass
    pass
                print(f"   {param}: {importance:.4f}")

        # Best Parameters
        print("\\\n🏆 BEST PARAMETERS:")
        best_params = analysis["best_parameters"]

        # Group parameters by category
        categories: dict[str, list[str]] = {
            "Strength Score Weights": [k for k in best_params if "weight" in k],
            "Level Detection": [
                k
                for k in best_params
                if k
                in [
                    "min_touch_count",
                    "min_level_age_hours",
                    "price_tolerance_pct",
                    "volume_threshold",
                    "strength_threshold",
                ]
            ],
            "Breakout Thresholds": [
                k
                for k in best_params
                if k
                in [
                    "breakout_threshold",
                    "confirmation_periods",
                    "volume_confirmation",
                    "momentum_threshold",
                    "false_breakout_filter",
                ]
            ],
            "Zone Multipliers": [
                k
                for k in best_params
                if k
                in [
                    "support_zone_multiplier",
                    "resistance_zone_multiplier",
                    "sr_zone_threshold",
                    "zone_expansion_factor",
                    "zone_contraction_factor",
                ]
            ],
            "Confidence Thresholds": [
                k
                for k in best_params
                if k
                in [
                    "min_sr_confidence",
                    "high_confidence_threshold",
                    "confidence_decay_rate",
                    "regime_confidence_boost",
                    "ensemble_confidence_threshold",
                ]
            ],
        }

        for category, params in categories.items():
    pass
    pass
            if params:
    pass
    pass
                print(f"\\\n   {category}:")
                for param in params:
    pass
    pass
                    print(f"     {param}: {best_params[param]:.4f}")

        # Recommendations
        if analysis["recommendations"]:
    pass
    pass
            print("\\\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(analysis["recommendations"], 1):
    pass
    pass
                print(f"   {i}. {rec}")

        print("\\\n" + "=" * 80)


async def main() -> int:
    """Main function to run S/R parameter optimization."""
    parser = argparse.ArgumentParser(description="S/R Parameter Optimization")
    parser.add_argument("--symbol", default="ETHUSDT", help="Trading symbol")
    parser.add_argument("--exchange", default="BINANCE", help="Exchange name")
    parser.add_argument("--period", type=int, default=365, help="Data period in days")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--output-dir",
        default="optimization_results",
        help="Output directory",
    )
    parser.add_argument("--config", help="Path to configuration file")

    args = parser.parse_args()

    # Configuration
    config: dict[str, Any] = {
        "sr_optimization": {
            "multi_objective": True,
            "objectives": ["sharpe_ratio", "win_rate", "signal_clarity"],
            "objective_weights": {
                "sharpe_ratio": 0.4,
                "win_rate": 0.3,
                "signal_clarity": 0.3,
            },
            "n_trials": args.n_trials,
            "cv_folds": 5,
            "early_stopping_patience": 20,
            "subsample_fraction": 0.7,
            "max_overfitting_threshold": 0.1,
            "min_validation_score": 0.5,
            "regularization_penalty": 0.1,
        },
    }

    try:
        # Initialize runner
    except Exception as e:
        pass
    except Exception as e:
        pass
        runner = SROptimizationRunner(config)

        # Prepare data
        price_data, target_returns = runner.prepare_sample_data(n_samples=2000)

        # Run optimization
        await runner.run_optimization(
            price_data=price_data,
            target_returns=target_returns,
            n_trials=args.n_trials,
        )

        # Analyze results
        analysis = runner.analyze_results()

        # Print comprehensive report
        runner.print_comprehensive_report(analysis)

        # Create visualizations
        plots = runner.create_visualizations(args.output_dir)
        if plots:
    pass
    pass
            print("\\\n📊 Visualizations saved:")
            for plot_name, plot_path in plots.items():
    pass
    pass
                print(f"   {plot_name}: {plot_path}")

        # Export parameters
        param_file = runner.export_parameters(
            f"{args.output_dir}/optimized_sr_parameters.json",
        )
        print(f"\\\n💾 Parameters exported to: {param_file}")

        print("\\\n🎉 S/R Parameter Optimization completed successfully!")

    except Exception as e:  # noqa: BLE001
        print(f"❌ Error during optimization: {e}")
        return 1

    return 0


if __name__ == "__main__":
    pass
    pass
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
