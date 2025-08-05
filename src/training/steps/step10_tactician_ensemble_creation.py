# src/training/steps/step10_tactician_ensemble_creation.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np

from src.utils.logger import system_logger


class TacticianEnsembleCreationStep:
    """Step 10: Tactician Ensemble Creation using Simple Averaging of calibrated probabilities."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger

    async def initialize(self) -> None:
        """Initialize the tactician ensemble creation step."""
        try:
            self.logger.info("Initializing Tactician Ensemble Creation Step...")
            self.logger.info(
                "Tactician Ensemble Creation Step initialized successfully",
            )

        except Exception as e:
            self.logger.error(
                f"Error initializing Tactician Ensemble Creation Step: {e}",
            )
            raise

    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute tactician ensemble creation.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing ensemble creation results
        """
        try:
            self.logger.info("🔄 Executing Tactician Ensemble Creation...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load tactician models
            models_dir = f"{data_dir}/tactician_models"
            tactician_models = {}

            # Load all model files
            for model_file in os.listdir(models_dir):
                if model_file.endswith(".pkl"):
                    model_name = model_file.replace(".pkl", "")
                    model_path = os.path.join(models_dir, model_file)

                    with open(model_path, "rb") as f:
                        tactician_models[model_name] = pickle.load(f)

            if not tactician_models:
                raise ValueError(f"No tactician models found in {models_dir}")

            # Create ensemble using simple averaging
            ensemble_results = await self._create_tactician_ensemble(
                tactician_models,
                symbol,
                exchange,
            )

            # Save ensemble model
            ensemble_models_dir = f"{data_dir}/tactician_ensembles"
            os.makedirs(ensemble_models_dir, exist_ok=True)

            ensemble_file = (
                f"{ensemble_models_dir}/{exchange}_{symbol}_tactician_ensemble.pkl"
            )
            with open(ensemble_file, "wb") as f:
                pickle.dump(ensemble_results, f)

            # Save ensemble summary
            summary_file = (
                f"{data_dir}/{exchange}_{symbol}_tactician_ensemble_summary.json"
            )
            with open(summary_file, "w") as f:
                json.dump(ensemble_results, f, indent=2)

            self.logger.info(
                f"✅ Tactician ensemble creation completed. Results saved to {ensemble_models_dir}",
            )

            # Update pipeline state
            pipeline_state["tactician_ensemble"] = ensemble_results

            return {
                "tactician_ensemble": ensemble_results,
                "ensemble_file": ensemble_file,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS",
            }

        except Exception as e:
            self.logger.error(f"❌ Error in Tactician Ensemble Creation: {e}")
            return {"status": "FAILED", "error": str(e), "duration": 0.0}

    async def _create_tactician_ensemble(
        self,
        models: dict[str, Any],
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """
        Create tactician ensemble using simple averaging of calibrated probabilities.

        Args:
            models: Tactician models
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            Dict containing ensemble model
        """
        try:
            self.logger.info(
                f"Creating tactician ensemble for {symbol} on {exchange}...",
            )

            # Extract models and their accuracies
            model_list = []
            model_names = []
            accuracies = []

            for model_name, model_data in models.items():
                model_list.append(model_data["model"])
                model_names.append(model_name)
                accuracies.append(model_data.get("accuracy", 0))

            # Create different ensemble types
            ensemble_results = {}

            # 1. Simple Averaging Ensemble (main method)
            simple_averaging_ensemble = await self._create_simple_averaging_ensemble(
                model_list,
                model_names,
                accuracies,
                symbol,
                exchange,
            )
            ensemble_results["simple_averaging"] = simple_averaging_ensemble

            # 2. Weighted Averaging Ensemble
            weighted_averaging_ensemble = (
                await self._create_weighted_averaging_ensemble(
                    model_list,
                    model_names,
                    accuracies,
                    symbol,
                    exchange,
                )
            )
            ensemble_results["weighted_averaging"] = weighted_averaging_ensemble

            # 3. Voting Ensemble
            voting_ensemble = await self._create_voting_ensemble(
                model_list,
                model_names,
                symbol,
                exchange,
            )
            ensemble_results["voting"] = voting_ensemble

            # 4. Stacking Ensemble
            stacking_ensemble = await self._create_stacking_ensemble(
                model_list,
                model_names,
                symbol,
                exchange,
            )
            ensemble_results["stacking"] = stacking_ensemble

            return ensemble_results

        except Exception as e:
            self.logger.error(f"Error creating tactician ensemble: {e}")
            raise

    async def _create_simple_averaging_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        accuracies: list[float],
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """Create simple averaging ensemble."""
        try:
            # Create ensemble class for simple averaging
            class SimpleAveragingEnsemble:
                def __init__(self, models, model_names):
                    self.models = models
                    self.model_names = model_names

                def predict_proba(self, X):
                    # Get predictions from all models
                    predictions = []
                    for model in self.models:
                        try:
                            pred = model.predict_proba(X)
                            predictions.append(pred)
                        except Exception:
                            # If model doesn't support predict_proba, use predict
                            pred = model.predict(X)
                            # Convert to probability format
                            pred_proba = np.zeros(
                                (len(pred), 3),
                            )  # Assuming 3 classes: -1, 0, 1
                            for i, p in enumerate(pred):
                                pred_proba[i, p + 1] = (
                                    1.0  # +1 to convert from [-1,0,1] to [0,1,2]
                                )
                            predictions.append(pred_proba)

                    # Average the predictions
                    avg_pred = np.mean(predictions, axis=0)
                    return avg_pred

                def predict(self, X):
                    proba = self.predict_proba(X)
                    return np.argmax(proba, axis=1) - 1  # Convert back to [-1,0,1]

            # Create ensemble
            ensemble = SimpleAveragingEnsemble(models, model_names)

            ensemble_data = {
                "ensemble": ensemble,
                "ensemble_type": "SimpleAveraging",
                "base_models": model_names,
                "base_accuracies": accuracies,
                "symbol": symbol,
                "exchange": exchange,
                "creation_date": datetime.now().isoformat(),
                "averaging_method": "simple_mean",
            }

            return ensemble_data

        except Exception as e:
            self.logger.error(f"Error creating simple averaging ensemble: {e}")
            raise

    async def _create_weighted_averaging_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        accuracies: list[float],
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """Create weighted averaging ensemble."""
        try:
            # Calculate weights based on accuracies
            total_accuracy = sum(accuracies)
            weights = [acc / total_accuracy for acc in accuracies]

            # Create ensemble class for weighted averaging
            class WeightedAveragingEnsemble:
                def __init__(self, models, model_names, weights):
                    self.models = models
                    self.model_names = model_names
                    self.weights = weights

                def predict_proba(self, X):
                    # Get predictions from all models
                    predictions = []
                    for model in self.models:
                        try:
                            pred = model.predict_proba(X)
                            predictions.append(pred)
                        except Exception:
                            # If model doesn't support predict_proba, use predict
                            pred = model.predict(X)
                            # Convert to probability format
                            pred_proba = np.zeros(
                                (len(pred), 3),
                            )  # Assuming 3 classes: -1, 0, 1
                            for i, p in enumerate(pred):
                                pred_proba[i, p + 1] = (
                                    1.0  # +1 to convert from [-1,0,1] to [0,1,2]
                                )
                            predictions.append(pred_proba)

                    # Weighted average the predictions
                    weighted_pred = np.zeros_like(predictions[0])
                    for i, (pred, weight) in enumerate(
                        zip(predictions, self.weights, strict=False),
                    ):
                        weighted_pred += pred * weight

                    return weighted_pred

                def predict(self, X):
                    proba = self.predict_proba(X)
                    return np.argmax(proba, axis=1) - 1  # Convert back to [-1,0,1]

            # Create ensemble
            ensemble = WeightedAveragingEnsemble(models, model_names, weights)

            ensemble_data = {
                "ensemble": ensemble,
                "ensemble_type": "WeightedAveraging",
                "base_models": model_names,
                "base_accuracies": accuracies,
                "weights": weights,
                "symbol": symbol,
                "exchange": exchange,
                "creation_date": datetime.now().isoformat(),
                "averaging_method": "weighted_mean",
            }

            return ensemble_data

        except Exception as e:
            self.logger.error(f"Error creating weighted averaging ensemble: {e}")
            raise

    async def _create_voting_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """Create voting ensemble."""
        try:
            from sklearn.ensemble import VotingClassifier

            # Create voting classifier
            estimators = [
                (name, model) for name, model in zip(model_names, models, strict=False)
            ]
            voting_classifier = VotingClassifier(estimators=estimators, voting="soft")

            ensemble_data = {
                "ensemble": voting_classifier,
                "ensemble_type": "Voting",
                "base_models": model_names,
                "voting_method": "soft",
                "symbol": symbol,
                "exchange": exchange,
                "creation_date": datetime.now().isoformat(),
            }

            return ensemble_data

        except Exception as e:
            self.logger.error(f"Error creating voting ensemble: {e}")
            raise

    async def _create_stacking_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        symbol: str,
        exchange: str,
    ) -> dict[str, Any]:
        """Create stacking ensemble."""
        try:
            from sklearn.ensemble import StackingClassifier
            from sklearn.linear_model import LogisticRegression

            # Create stacking classifier
            estimators = [
                (name, model) for name, model in zip(model_names, models, strict=False)
            ]
            stacking_classifier = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                cv=5,
                stack_method="predict_proba",
            )

            ensemble_data = {
                "ensemble": stacking_classifier,
                "ensemble_type": "Stacking",
                "base_models": model_names,
                "final_estimator": "LogisticRegression",
                "cv_folds": 5,
                "symbol": symbol,
                "exchange": exchange,
                "creation_date": datetime.now().isoformat(),
            }

            return ensemble_data

        except Exception as e:
            self.logger.error(f"Error creating stacking ensemble: {e}")
            raise


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    **kwargs,
) -> bool:
    """
    Run the tactician ensemble creation step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create step instance
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = TacticianEnsembleCreationStep(config)
        await step.initialize()

        # Execute step
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            **kwargs,
        }

        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)

        return result.get("status") == "SUCCESS"

    except Exception as e:
        print(f"❌ Tactician ensemble creation failed: {e}")
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
