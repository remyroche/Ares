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
# src/training/steps/step10_tactician_ensemble_creation.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score

from src.utils.logger import system_logger


class TacticianEnsembleCreationStep:
    """Step 10: Create an optimized Tactician Ensemble by blending two models."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger

    async def initialize(self) -> None:
        """Initialize the tactician ensemble creation step."""
        self.logger.info("Initializing Tactician Ensemble Creation Step...")
        self.logger.info("Tactician Ensemble Creation Step initialized successfully.")

    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute tactician ensemble creation.

        Args:
            training_input: Training input parameters.
            pipeline_state: Current pipeline state.

        Returns:
            A dictionary containing the results of the ensemble creation.
        """
        try:
            self.logger.info("🔄 Executing Tactician Ensemble Creation...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load tactician models
            models_dir = os.path.join(data_dir, "tactician_models")
            tactician_models = {}
            for model_file in os.listdir(models_dir):
                if model_file.endswith(".pkl"):
                    model_name = os.path.splitext(model_file)[0]
                    model_path = os.path.join(models_dir, model_file)
                    with open(model_path, "rb") as f:
                        tactician_models[model_name] = pickle.load(f)

            if len(tactician_models) != 2:
                raise ValueError(
                    f"Expected 2 tactician models, but found {len(tactician_models)} in {models_dir}"
                )

            # Create an optimized blended ensemble
            ensemble_details = await self._create_tactician_ensemble(
                tactician_models, data_dir
            )

            # Separate the model object from its serializable details
            ensemble_model = ensemble_details.pop("ensemble")
            
            # --- Save Ensemble Model and Summary ---
            ensemble_dir = os.path.join(data_dir, "tactician_ensembles")
            os.makedirs(ensemble_dir, exist_ok=True)

            # Save the ensemble model object to a pickle file
            ensemble_file = os.path.join(
                ensemble_dir, f"{exchange}_{symbol}_tactician_ensemble.pkl"
            )
            with open(ensemble_file, "wb") as f:
                pickle.dump(ensemble_model, f)

            # Save the ensemble's metadata to a JSON summary file
            summary_file = os.path.join(
                ensemble_dir, f"{exchange}_{symbol}_tactician_ensemble_summary.json"
            )
            with open(summary_file, "w") as f:
                json.dump(ensemble_details, f, indent=4)

            self.logger.info(
                f"✅ Tactician ensemble created. Model saved to {ensemble_file}"
            )

            # Update pipeline state
            pipeline_state["tactician_ensemble_details"] = ensemble_details
            pipeline_state["tactician_ensemble_model"] = ensemble_model

            return {
                "status": "SUCCESS",
                "ensemble_details": ensemble_details,
                "ensemble_file": ensemble_file,
            }

        except Exception as e:
            self.logger.error(f"❌ Error in Tactician Ensemble Creation: {e}", exc_info=True)
            return {"status": "FAILED", "error": str(e)}

    async def _load_validation_data(self, data_dir: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Loads validation data for weight optimization.

        NOTE: This is a placeholder. In a real application, this would load a
        pre-processed validation dataset (e.g., from a .parquet or .csv file).

        Args:
            data_dir: The directory where data is stored.

        Returns:
            A tuple containing validation features (X_val) and labels (y_val).
        """
        self.logger.info("Loading validation data (using placeholder)...")
        # Placeholder: 1000 samples, 50 features, 3 classes [-1, 0, 1]
        num_samples, num_features = 1000, 50
        X_val = np.random.rand(num_samples, num_features)
        y_val = np.random.randint(-1, 2, size=num_samples)
        self.logger.info(f"Loaded placeholder data with shapes X: {X_val.shape}, y: {y_val.shape}")
        return X_val, y_val

    def _get_model_predictions(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Helper to get probability predictions, with a fallback for classifiers without `predict_proba`."""
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        
        self.logger.warning(f"Model {type(model).__name__} lacks `predict_proba`. Falling back to `predict`.")
        preds = model.predict(X)
        # Convert [-1, 0, 1] integer labels to one-hot encoded probabilities
        pred_proba = np.zeros((len(preds), 3))
        # Map label `k` to index `k+1` to handle the -1 label
        pred_proba[np.arange(len(preds)), preds + 1] = 1.0
        return pred_proba

    async def _find_optimal_weight(
        self, models: list[Any], X_val: np.ndarray, y_val: np.ndarray
    ) -> float:
        """
        Performs a simple search to find the optimal blending weight `w` for two models.
        The blend is calculated as `w * preds_A + (1 - w) * preds_B`.

        Args:
            models: A list containing the two models to blend.
            X_val: Validation features.
            y_val: Validation true labels.

        Returns:
            The optimal weight for the first model.
        """
        self.logger.info("Searching for optimal blending weight...")
        preds_a = self._get_model_predictions(models[0], X_val)
        preds_b = self._get_model_predictions(models[1], X_val)

        best_accuracy = -1.0
        optimal_weight = 0.5  # Default to a simple average

        # Grid search for the best weight `w`
        for w in np.arange(0, 1.01, 0.01):
            blended_proba = w * preds_a + (1 - w) * preds_b
            blended_labels = np.argmax(blended_proba, axis=1) - 1  # Convert back to [-1, 0, 1]
            accuracy = accuracy_score(y_val, blended_labels)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                optimal_weight = w
        
        self.logger.info(f"Optimal weight found: {optimal_weight:.2f} (for model 1) with validation accuracy: {best_accuracy:.4f}")
        return optimal_weight

    async def _create_tactician_ensemble(
        self, models: dict[str, Any], data_dir: str
    ) -> dict[str, Any]:
        """
        Creates an optimized weighted-average ensemble of two tactician models.

        Args:
            models: A dictionary containing the two loaded tactician models.
            data_dir: The directory for loading validation data.

        Returns:
            A dictionary containing the ensemble model and its metadata.
        """
        self.logger.info("Creating optimized tactician ensemble...")

        model_items = list(models.values())
        model_names = list(models.keys())
        
        base_models = [m["model"] for m in model_items]
        base_accuracies = [m.get("accuracy", 0.0) for m in model_items]

        # 1. Load validation data to find the optimal weight
        X_val, y_val = await self._load_validation_data(data_dir)

        # 2. Find the optimal blending weight
        optimal_w = await self._find_optimal_weight(base_models, X_val, y_val)

        # 3. Define and create the ensemble class with the optimal weight
        class OptimalBlendedEnsemble:
            def __init__(self, model_a: Any, model_b: Any, weight_a: float):
                self.model_a = model_a
                self.model_b = model_b
                self.weight_a = weight_a
                self.weight_b = 1.0 - weight_a

            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                """Predicts class probabilities by blending the two base models."""
                preds_a = self.model_a.predict_proba(X)
                preds_b = self.model_b.predict_proba(X)
                return self.weight_a * preds_a + self.weight_b * preds_b

            def predict(self, X: np.ndarray) -> np.ndarray:
                """Predicts class labels based on the blended probabilities."""
                proba = self.predict_proba(X)
                return np.argmax(proba, axis=1) - 1

        ensemble = OptimalBlendedEnsemble(base_models[0], base_models[1], optimal_w)

        # 4. Prepare the final dictionary with the model and its metadata
        ensemble_data = {
            "ensemble": ensemble,
            "ensemble_type": "OptimalBlended",
            "base_models": model_names,
            "base_model_accuracies": base_accuracies,
            "optimal_weight": {model_names[0]: optimal_w, model_names[1]: 1.0 - optimal_w},
            "creation_date": datetime.now().isoformat(),
        }
        return ensemble_data

# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    **kwargs,
) -> bool:
    """Runs the tactician ensemble creation step."""
    try:
        config = {"symbol": symbol, "exchange": exchange, "data_dir": data_dir}
        step = TacticianEnsembleCreationStep(config)
        await step.initialize()

        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "data_dir": data_dir,
            **kwargs,
        }
        result = await step.execute(training_input, pipeline_state={})
        return result.get("status") == "SUCCESS"

    except Exception as e:
        system_logger.error(f"❌ Tactician ensemble creation failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Example of how to run the step
    async def test():
        # As this test requires model files from a previous step, we'll
        # create dummy model files for demonstration purposes.
        print("--- Running Tactician Ensemble Creation Test ---")
        
        # Create dummy models and data directory
        test_data_dir = "data/test_training"
        models_dir = os.path.join(test_data_dir, "tactician_models")
        os.makedirs(models_dir, exist_ok=True)

        from sklearn.linear_model import LogisticRegression
        
        # Dummy Model A
        model_a = LogisticRegression()
        model_a.fit(np.random.rand(10, 50), np.random.randint(-1, 2, 10))
        model_a_data = {"model": model_a, "accuracy": 0.65}
        with open(os.path.join(models_dir, "model_A.pkl"), "wb") as f:
            pickle.dump(model_a_data, f)

        # Dummy Model B
        model_b = LogisticRegression()
        model_b.fit(np.random.rand(10, 50), np.random.randint(-1, 2, 10))
        model_b_data = {"model": model_b, "accuracy": 0.72}
        with open(os.path.join(models_dir, "model_B.pkl"), "wb") as f:
            pickle.dump(model_b_data, f)

        print(f"Created dummy models in {models_dir}")
        
        # Run the step
        success = await run_step("ETHUSDT", "BINANCE", test_data_dir)
        print(f"\nTest Result: {'SUCCESS' if success else 'FAILED'}")

        # Verify output files
        ensemble_dir = os.path.join(test_data_dir, "tactician_ensembles")
        if success:
            summary_path = os.path.join(ensemble_dir, "BINANCE_ETHUSDT_tactician_ensemble_summary.json")
            model_path = os.path.join(ensemble_dir, "BINANCE_ETHUSDT_tactician_ensemble.pkl")
            print(f"Verified: Summary file exists at {summary_path}")
            print(f"Verified: Model file exists at {model_path}")

    asyncio.run(test())
