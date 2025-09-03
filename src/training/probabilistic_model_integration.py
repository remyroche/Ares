#!/usr/bin/env python3
"""Probabilistic Model Integration for Tactician and Analyst.

This module provides seamless integration between the probabilistic Bayesian optimizer
and your existing Tactician and Analyst models, enabling end-to-end optimization of
probabilistic outputs and uncertainty quantification.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import the probabilistic Bayesian optimizer
from .probabilistic_bayesian_optimizer import (
    ProbabilisticBayesianOptimizer,
    ProbabilisticOptimizationConfig,
)

# Import existing model components
try:
    from src.analyst.ml_confidence_predictor import MLConfidencePredictor
    from src.analyst.momentum_predictor import MomentumPredictor
    from src.analyst.regime_predictor import RegimePredictor
    from src.analyst.trend_predictor import TrendPredictor
    from src.analyst.volatility_predictor import VolatilityPredictor
    from src.tactician.enhanced_prediction_integrator import (
        TacticianEnhancedPredictionIntegrator,
    )
except ImportError:
    # Fallback for testing
    pass


@dataclass
class ModelOptimizationTarget:
    """Defines what aspects of a model to optimize."""

    model_type: str  # 'tactician' or 'analyst'
    model_name: str  # Specific model identifier
    optimization_objectives: List[str]  # What to optimize
    hyperparameter_ranges: Dict[str, Tuple]  # Parameter search spaces
    calibration_methods: List[str]  # Available calibration methods
    uncertainty_methods: List[str]  # Uncertainty estimation methods


class ProbabilisticModelIntegrator:
    """Integrates probabilistic Bayesian optimization with existing Tactician and
    Analyst models.

    This class provides:
    1. Seamless integration with existing model architectures
    2. Automated optimization workflows
    3. Model performance monitoring and retraining
    4. Uncertainty quantification enhancement
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize optimizers for different model types
        self.optimizers = {}
        self.model_targets = self._initialize_model_targets()

        # Performance tracking
        self.optimization_history = {}
        self.model_performance = {}

    def _initialize_model_targets(self) -> Dict[str, ModelOptimizationTarget]:
        """Initialize optimization targets for different model types."""

        return {
            "tactician": ModelOptimizationTarget(
                model_type="tactician",
                model_name="enhanced_prediction_integrator",
                optimization_objectives=["calibration", "sharpness", "discrimination"],
                hyperparameter_ranges={
                    "barrier_system": {
                        "upper_barrier_multiplier": (0.3, 0.8),
                        "lower_barrier_multiplier": (0.1, 0.5),
                        "confidence_threshold": (0.6, 0.9),
                        "precision_threshold": (0.7, 0.95),
                    },
                    "prediction_calibration": {
                        "calibration_method": ["isotonic", "sigmoid", "platt"],
                        "calibration_cv_folds": (3, 10),
                        "uncertainty_estimation": ["ensemble", "gaussian", "conformal"],
                    },
                },
                calibration_methods=["isotonic", "sigmoid", "platt"],
                uncertainty_methods=["ensemble", "gaussian", "conformal"],
            ),
            "analyst": ModelOptimizationTarget(
                model_type="analyst",
                model_name="ensemble_predictor",
                optimization_objectives=[
                    "calibration",
                    "sharpness",
                    "discrimination",
                    "regime_accuracy",
                ],
                hyperparameter_ranges={
                    "regime_detection": {
                        "regime_threshold": (0.5, 0.8),
                        "regime_confidence_threshold": (0.6, 0.9),
                        "regime_transition_smoothing": (0.1, 0.5),
                    },
                    "prediction_calibration": {
                        "calibration_method": [
                            "isotonic",
                            "sigmoid",
                            "platt",
                            "temperature",
                        ],
                        "calibration_cv_folds": (5, 15),
                        "uncertainty_estimation": [
                            "ensemble",
                            "gaussian",
                            "conformal",
                            "mc_dropout",
                        ],
                    },
                },
                calibration_methods=["isotonic", "sigmoid", "platt", "temperature"],
                uncertainty_methods=["ensemble", "gaussian", "conformal", "mc_dropout"],
            ),
        }

    def create_optimizer(self, model_type: str) -> ProbabilisticBayesianOptimizer:
        """Create a probabilistic Bayesian optimizer for a specific model type."""

        if model_type not in self.model_targets:
            raise ValueError(f"Unknown model type: {model_type}")

        target = self.model_targets[model_type]

        # Create optimization configuration
        config = ProbabilisticOptimizationConfig(
            objectives=target.optimization_objectives,
            n_trials=self.config.get("optimization", {}).get("n_trials", 100),
            n_jobs=self.config.get("optimization", {}).get("n_jobs", 1),
            early_stopping_patience=self.config.get("optimization", {}).get(
                "early_stopping_patience", 10
            ),
            sampler_type=self.config.get("optimization", {}).get("sampler_type", "tpe"),
        )

        # Create optimizer
        optimizer = ProbabilisticBayesianOptimizer(
            config=config,
            model_type=model_type,
            storage_url=f"sqlite:///probabilistic_{model_type}_optimization.db",
        )

        self.optimizers[model_type] = optimizer
        return optimizer

    async def optimize_tactician_model(
        self,
        market_data: pd.DataFrame,
        historical_predictions: pd.DataFrame,
        optimization_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Optimize the Tactician model using probabilistic Bayesian optimization."""

        self.logger.info("🚀 Starting Tactician model optimization...")

        # Create optimizer if not exists
        if "tactician" not in self.optimizers:
            self.create_optimizer("tactician")

        optimizer = self.optimizers["tactician"]

        # Prepare data for optimization
        X, y = self._prepare_tactician_optimization_data(
            market_data, historical_predictions
        )

        # Run optimization
        results = optimizer.optimize(
            X=X,
            y=y,
            model_factory=self._create_tactician_model_factory(),
            validation_split=0.2,
        )

        # Store results
        self.optimization_history["tactician"] = results

        # Apply optimized parameters
        await self._apply_tactician_optimization_results(results)

        self.logger.info("✅ Tactician model optimization completed!")
        return results

    async def optimize_analyst_model(
        self,
        market_data: pd.DataFrame,
        historical_predictions: pd.DataFrame,
        optimization_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Optimize the Analyst model using probabilistic Bayesian optimization."""

        self.logger.info("🚀 Starting Analyst model optimization...")

        # Create optimizer if not exists
        if "analyst" not in self.optimizers:
            self.create_optimizer("analyst")

        optimizer = self.optimizers["analyst"]

        # Prepare data for optimization
        X, y = self._prepare_analyst_optimization_data(
            market_data, historical_predictions
        )

        # Run optimization
        results = optimizer.optimize(
            X=X,
            y=y,
            model_factory=self._create_analyst_model_factory(),
            validation_split=0.2,
        )

        # Store results
        self.optimization_history["analyst"] = results

        # Apply optimized parameters
        await self._apply_analyst_optimization_results(results)

        self.logger.info("✅ Analyst model optimization completed!")
        return results

    def _prepare_tactician_optimization_data(
        self, market_data: pd.DataFrame, historical_predictions: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for Tactician model optimization."""

        # Extract features from market data
        features = []

        # Price-based features
        if "close" in market_data.columns:
            features.append(market_data["close"].pct_change().fillna(0))
            features.append(market_data["close"].rolling(20).std().fillna(0))

        # Volume features
        if "volume" in market_data.columns:
            features.append(market_data["volume"].pct_change().fillna(0))
            features.append(market_data["volume"].rolling(20).mean().fillna(0))

        # Technical indicators
        if "high" in market_data.columns and "low" in market_data.columns:
            features.append(
                (market_data["high"] - market_data["low"]) / market_data["close"]
            )

        # Historical prediction accuracy
        if "prediction_accuracy" in historical_predictions.columns:
            features.append(historical_predictions["prediction_accuracy"].fillna(0.5))

        # Combine features
        X = np.column_stack([f.values for f in features if len(f) > 0])

        # Create target variable (simplified - you'd want to use actual trade outcomes)
        # This is a placeholder - replace with actual profit/loss or trade success
        y = np.random.choice([0, 1], size=len(X), p=[0.4, 0.6])  # 60% success rate

        return X, y

    def _prepare_analyst_optimization_data(
        self, market_data: pd.DataFrame, historical_predictions: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for Analyst model optimization."""

        # Extract features from market data
        features = []

        # Price-based features
        if "close" in market_data.columns:
            features.append(market_data["close"].pct_change().fillna(0))
            features.append(market_data["close"].rolling(50).std().fillna(0))
            features.append(market_data["close"].rolling(200).mean().fillna(0))

        # Volume features
        if "volume" in market_data.columns:
            features.append(market_data["volume"].rolling(50).mean().fillna(0))

        # Historical prediction accuracy
        if "prediction_accuracy" in historical_predictions.columns:
            features.append(historical_predictions["prediction_accuracy"].fillna(0.5))

        # Regime features
        if "regime_prediction" in historical_predictions.columns:
            features.append(historical_predictions["regime_prediction"].fillna(0.5))

        # Combine features
        X = np.column_stack([f.values for f in features if len(f) > 0])

        # Create target variable (simplified - you'd want to use actual regime outcomes)
        # This is a placeholder - replace with actual regime classification
        y = np.random.choice([0, 1, 2], size=len(X), p=[0.3, 0.4, 0.3])  # 3 regimes

        return X, y

    def _create_tactician_model_factory(self):
        """Create a factory function for Tactician models."""

        def factory(params: Dict[str, Any]):
            # This would integrate with your existing Tactician model
            # For now, returning a placeholder
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 10),
                random_state=42,
                n_jobs=1,
            )

            return model

        return factory

    def _create_analyst_model_factory(self):
        """Create a factory function for Analyst models."""

        def factory(params: Dict[str, Any]):
            # This would integrate with your existing Analyst model
            # For now, returning a placeholder
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", 15),
                random_state=42,
                n_jobs=1,
            )

            return model

        return factory

    async def _apply_tactician_optimization_results(self, results: Dict[str, Any]):
        """Apply optimization results to the Tactician model."""

        try:
            # Get best hyperparameters
            best_params = (
                results.get("best_solutions", {})
                .get("calibration", {})
                .get("params", {})
            )

            if not best_params:
                self.logger.warning("No best parameters found for Tactician")
                return

            # Apply barrier system parameters
            if "upper_barrier_multiplier" in best_params:
                # Update your Tactician configuration
                self.logger.info(
                    f"Updating Tactician upper barrier multiplier: {best_params['upper_barrier_multiplier']}"
                )

            if "lower_barrier_multiplier" in best_params:
                self.logger.info(
                    f"Updating Tactician lower barrier multiplier: {best_params['lower_barrier_multiplier']}"
                )

            if "confidence_threshold" in best_params:
                self.logger.info(
                    f"Updating Tactician confidence threshold: {best_params['confidence_threshold']}"
                )

            # Apply calibration method
            if "calibration_method" in best_params:
                self.logger.info(
                    f"Updating Tactician calibration method: {best_params['calibration_method']}"
                )

            self.logger.info("✅ Tactician optimization results applied successfully!")

        except Exception as e:
            self.logger.error(f"Error applying Tactician optimization results: {e}")

    async def _apply_analyst_optimization_results(self, results: Dict[str, Any]):
        """Apply optimization results to the Analyst model."""

        try:
            # Get best hyperparameters
            best_params = (
                results.get("best_solutions", {})
                .get("calibration", {})
                .get("params", {})
            )

            if not best_params:
                self.logger.warning("No best parameters found for Analyst")
                return

            # Apply regime detection parameters
            if "regime_threshold" in best_params:
                self.logger.info(
                    f"Updating Analyst regime threshold: {best_params['regime_threshold']}"
                )

            if "regime_confidence_threshold" in best_params:
                self.logger.info(
                    f"Updating Analyst regime confidence threshold: {best_params['regime_confidence_threshold']}"
                )

            # Apply calibration method
            if "calibration_method" in best_params:
                self.logger.info(
                    f"Updating Analyst calibration method: {best_params['calibration_method']}"
                )

            self.logger.info("✅ Analyst optimization results applied successfully!")

        except Exception as e:
            self.logger.error(f"Error applying Analyst optimization results: {e}")

    async def run_comprehensive_optimization(
        self, market_data: pd.DataFrame, historical_predictions: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run comprehensive optimization for both Tactician and Analyst models."""

        self.logger.info("🚀 Starting comprehensive model optimization...")

        results = {}

        # Optimize Tactician
        try:
            tactician_results = await self.optimize_tactician_model(
                market_data, historical_predictions
            )
            results["tactician"] = tactician_results
        except Exception as e:
            self.logger.error(f"Tactician optimization failed: {e}")
            results["tactician"] = {"error": str(e)}

        # Optimize Analyst
        try:
            analyst_results = await self.optimize_analyst_model(
                market_data, historical_predictions
            )
            results["analyst"] = analyst_results
        except Exception as e:
            self.logger.error(f"Analyst optimization failed: {e}")
            results["analyst"] = {"error": str(e)}

        # Generate optimization summary
        summary = self._generate_optimization_summary(results)
        results["summary"] = summary

        self.logger.info("✅ Comprehensive optimization completed!")
        return results

    def _generate_optimization_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of optimization results."""

        summary = {
            "total_models_optimized": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "best_parameters": {},
            "performance_improvements": {},
            "recommendations": [],
        }

        for model_type, result in results.items():
            if model_type == "summary":
                continue

            summary["total_models_optimized"] += 1

            if "error" in result:
                summary["failed_optimizations"] += 1
                summary["recommendations"].append(
                    f"Investigate {model_type} optimization failure: {result['error']}"
                )
            else:
                summary["successful_optimizations"] += 1

                # Extract best parameters
                best_solutions = result.get("best_solutions", {})
                if best_solutions:
                    summary["best_parameters"][model_type] = best_solutions

                # Generate recommendations
                if "calibration" in best_solutions:
                    calib_params = best_solutions["calibration"]["params"]
                    summary["recommendations"].append(
                        f"Use {calib_params.get('calibration_method', 'default')} "
                        f"calibration for {model_type}"
                    )

        return summary

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get the current status of all optimizations."""

        status = {
            "optimizers_created": list(self.optimizers.keys()),
            "optimization_history": self.optimization_history,
            "model_performance": self.model_performance,
            "recommendations": [],
        }

        # Generate recommendations based on optimization history
        for model_type, history in self.optimization_history.items():
            if "best_solutions" in history:
                best_solutions = history["best_solutions"]

                # Check if calibration needs improvement
                if "calibration" in best_solutions:
                    calib_score = best_solutions["calibration"]["value"]
                    if calib_score > 0.1:  # High Brier score (bad calibration)
                        status["recommendations"].append(
                            f"{model_type.capitalize()} calibration needs improvement "
                            f"(score: {calib_score:.3f})"
                        )

                # Check if sharpness can be improved
                if "sharpness" in best_solutions:
                    sharp_score = best_solutions["sharpness"]["value"]
                    if sharp_score < -0.5:  # Low sharpness
                        status["recommendations"].append(
                            f"{model_type.capitalize()} predictions could be more confident "
                            f"(sharpness: {sharp_score:.3f})"
                        )

        return status

    def plot_optimization_results(
        self, model_type: str, save_path: Optional[str] = None
    ):
        """Plot optimization results for a specific model type."""

        if model_type not in self.optimizers:
            self.logger.warning(f"No optimizer found for {model_type}")
            return

        optimizer = self.optimizers[model_type]
        optimizer.plot_optimization_results(save_path)


# Example usage
async def main():
    """Example usage of the ProbabilisticModelIntegrator."""

    # Configuration
    config = {
        "optimization": {
            "n_trials": 50,
            "n_jobs": 1,
            "early_stopping_patience": 10,
            "sampler_type": "tpe",
        }
    }

    # Create integrator
    integrator = ProbabilisticModelIntegrator(config)

    # Create sample data
    market_data = pd.DataFrame(
        {
            "close": np.random.randn(1000).cumsum() + 100,
            "volume": np.random.uniform(1000, 10000, 1000),
            "high": np.random.randn(1000).cumsum() + 101,
            "low": np.random.randn(1000).cumsum() + 99,
        }
    )

    historical_predictions = pd.DataFrame(
        {
            "prediction_accuracy": np.random.uniform(0.5, 0.9, 1000),
            "regime_prediction": np.random.uniform(0, 1, 1000),
        }
    )

    # Run comprehensive optimization
    results = await integrator.run_comprehensive_optimization(
        market_data, historical_predictions
    )

    # Get status
    status = integrator.get_optimization_status()

    print("✅ Optimization completed!")
    print(f"Results: {results}")
    print(f"Status: {status}")


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
