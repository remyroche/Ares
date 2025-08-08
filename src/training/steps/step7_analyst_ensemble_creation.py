# src/training/steps/step7_analyst_ensemble_creation.py

import asyncio
import json
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.utils.logger import system_logger
from src.analyst.data_utils import load_klines_data

try:
    import joblib  # Optional; used for .joblib artifacts
except Exception:  # pragma: no cover
    joblib = None


class DynamicWeightedEnsemble:
    """Dynamic weighted ensemble using Sharpe Ratio for model weighting."""
    
    def __init__(self, models, model_names, weights):
        self.models = models
        self.model_names = model_names
        self.weights = weights
    
    def predict(self, X):
        """Make ensemble predictions using weighted average of model probabilities."""
        all_probabilities = []
        
        for name, model in zip(self.model_names, self.models):
            if self.weights.get(name, 0) > 0:
                try:
                    probs = model.predict_proba(X)
                    weighted_probs = probs * self.weights.get(name, 0)
                    all_probabilities.append(weighted_probs)
                except Exception:
                    continue
        
        if all_probabilities:
            # Average the weighted probabilities
            ensemble_probs = np.mean(all_probabilities, axis=0)
            return np.argmax(ensemble_probs, axis=1)
        else:
            # Fallback: return random predictions
            return np.random.randint(0, 2, size=len(X))
    
    def predict_proba(self, X):
        """Get ensemble probability predictions."""
        all_probabilities = []
        
        for name, model in zip(self.model_names, self.models):
            if self.weights.get(name, 0) > 0:
                try:
                    probs = model.predict_proba(X)
                    weighted_probs = probs * self.weights.get(name, 0)
                    all_probabilities.append(weighted_probs)
                except Exception:
                    continue
        
        if all_probabilities:
            # Average the weighted probabilities
            return np.mean(all_probabilities, axis=0)
        else:
            # Fallback: return uniform probabilities
            return np.ones((len(X), 2)) * 0.5


class AnalystEnsembleCreationStep:
    """Step 7: Analyst Ensemble Creation using StackingCV and Dynamic Weighting."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger

    async def initialize(self) -> None:
        """Initialize the analyst ensemble creation step."""
        try:
            self.logger.info("Initializing Analyst Ensemble Creation Step...")
            self.logger.info("Analyst Ensemble Creation Step initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing Analyst Ensemble Creation Step: {e}")
            raise

    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute analyst ensemble creation with data loading and proper ensemble methods.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing ensemble creation results
        """
        try:
            self.logger.info("🔄 Executing Analyst Ensemble Creation...")

            # Extract parameters
            symbol = training_input.get("symbol", "ETHUSDT")
            exchange = training_input.get("exchange", "BINANCE")
            data_dir = training_input.get("data_dir", "data/training")

            # Load training and validation data
            training_data, validation_data = await self._load_training_data(
                symbol, exchange, data_dir
            )

            if training_data is None or validation_data is None:
                raise ValueError("Failed to load training and validation data")

            # Load enhanced analyst models
            enhanced_models_dir = f"{data_dir}/enhanced_analyst_models"
            regime_dirs = [d for d in os.listdir(enhanced_models_dir) if os.path.isdir(os.path.join(enhanced_models_dir, d))]

            if not regime_dirs:
                raise ValueError(
                    f"No enhanced analyst models found in {enhanced_models_dir}",
                )

            # Create ensembles for each regime
            ensemble_results = {}

            for regime_name in regime_dirs:
                self.logger.info(f"Creating ensemble for regime: {regime_name}")

                # Get regime-specific data
                regime_training_data = training_data.get(regime_name, pd.DataFrame())
                regime_validation_data = validation_data.get(regime_name, pd.DataFrame())

                if regime_training_data.empty or regime_validation_data.empty:
                    self.logger.warning(f"No data available for regime: {regime_name}")
                    continue

                # Lazy-load models for this regime only
                regime_path = os.path.join(enhanced_models_dir, regime_name)
                regime_models = {}
                for model_file in os.listdir(regime_path):
                    if model_file.endswith((".pkl", ".joblib")):
                        model_name = model_file.replace(".pkl", "").replace(".joblib", "")
                        model_path = os.path.join(regime_path, model_file)

                        if model_file.endswith(".joblib") and joblib is not None:
                            regime_models[model_name] = joblib.load(model_path)
                        else:
                            with open(model_path, "rb") as f:
                                regime_models[model_name] = pickle.load(f)

                # Create ensemble for this regime
                regime_ensemble = await self._create_regime_ensemble(
                    regime_models,
                    regime_name,
                    regime_training_data,
                    regime_validation_data,
                )
                ensemble_results[regime_name] = regime_ensemble
                # Free models from memory before next regime
                regime_models.clear()
            
            # Save ensemble models
            ensemble_models_dir = f"{data_dir}/analyst_ensembles"
            os.makedirs(ensemble_models_dir, exist_ok=True)

            for regime_name, ensemble_data in ensemble_results.items():
                ensemble_file = f"{ensemble_models_dir}/{regime_name}_ensemble.pkl"
                with open(ensemble_file, "wb") as f:
                    pickle.dump(ensemble_data, f)

            # Save ensemble summary (without ensemble objects for JSON serialization)
            summary_file = (
                f"{data_dir}/{exchange}_{symbol}_analyst_ensemble_summary.json"
            )
            
            # Create JSON-serializable summary
            json_summary = {}
            for regime_name, regime_ensembles in ensemble_results.items():
                json_summary[regime_name] = {}
                for ensemble_type, ensemble_data in regime_ensembles.items():
                    # Extract only JSON-serializable data
                    json_summary[regime_name][ensemble_type] = {
                        "ensemble_type": ensemble_data.get("ensemble_type"),
                        "base_models": ensemble_data.get("base_models"),
                        "regime": ensemble_data.get("regime"),
                        "creation_date": ensemble_data.get("creation_date"),
                        "validation_metrics": ensemble_data.get("validation_metrics"),
                        "cv_scores": ensemble_data.get("cv_scores"),
                        "weights": ensemble_data.get("weights"),
                        "sharpe_ratios": ensemble_data.get("sharpe_ratios"),
                        "features": ensemble_data.get("features"),
                    }
            
            with open(summary_file, "w") as f:
                json.dump(json_summary, f, indent=2)

            self.logger.info(
                f"✅ Analyst ensemble creation completed. Results saved to {ensemble_models_dir}",
            )

            # Update pipeline state
            pipeline_state["analyst_ensembles"] = ensemble_results

            return {
                "analyst_ensembles": ensemble_results,
                "ensemble_models_dir": ensemble_models_dir,
                "duration": 0.0,  # Will be calculated in actual implementation
                "status": "SUCCESS",
            }

        except Exception as e:
            self.logger.error(f"❌ Error in Analyst Ensemble Creation: {e}")
            return {"status": "FAILED", "error": str(e), "duration": 0.0}

    async def _load_training_data(
        self, symbol: str, exchange: str, data_dir: str
    ) -> tuple[dict[str, pd.DataFrame] | None, dict[str, pd.DataFrame] | None]:
        """
        Load training and validation data for all regimes.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory path

        Returns:
            Tuple of (training_data, validation_data) dictionaries keyed by regime
        """
        try:
            self.logger.info("Loading training and validation data...")

            # Load regime data files
            regime_data_dir = f"{data_dir}/regime_data"
            if not os.path.exists(regime_data_dir):
                self.logger.error(f"Regime data directory not found: {regime_data_dir}")
                return None, None

            training_data = {}
            validation_data = {}

            for regime_file in os.listdir(regime_data_dir):
                if regime_file.endswith("_training.csv"):
                    regime_name = regime_file.replace("_training.csv", "")
                    training_file = os.path.join(regime_data_dir, regime_file)
                    validation_file = os.path.join(regime_data_dir, f"{regime_name}_validation.csv")

                    # Load training data
                    if os.path.exists(training_file):
                        training_df = pd.read_csv(training_file, index_col=0, parse_dates=True)
                        training_data[regime_name] = training_df
                        self.logger.info(f"Loaded training data for {regime_name}: {training_df.shape}")

                    # Load validation data
                    if os.path.exists(validation_file):
                        validation_df = pd.read_csv(validation_file, index_col=0, parse_dates=True)
                        validation_data[regime_name] = validation_df
                        self.logger.info(f"Loaded validation data for {regime_name}: {validation_df.shape}")

            if not training_data or not validation_data:
                self.logger.warning("No training or validation data found, attempting to load from klines data...")
                
                # Fallback: try to load from klines data and split
                klines_file = f"{data_dir}/{exchange}_{symbol}_1h.csv"
                if os.path.exists(klines_file):
                    klines_df = load_klines_data(klines_file)
                    if not klines_df.empty:
                        # Split data into training and validation
                        split_idx = int(len(klines_df) * 0.8)
                        training_data["all"] = klines_df.iloc[:split_idx]
                        validation_data["all"] = klines_df.iloc[split_idx:]
                        self.logger.info(f"Split klines data: training={len(training_data['all'])}, validation={len(validation_data['all'])}")

            return training_data, validation_data

        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            return None, None

    async def _create_regime_ensemble(
        self,
        models: dict[str, Any],
        regime_name: str,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Create ensemble for a specific regime with real data.

        Args:
            models: Enhanced models for the regime
            regime_name: Name of the regime
            training_data: Training data for the regime
            validation_data: Validation data for the regime

        Returns:
            Dict containing ensemble model
        """
        try:
            self.logger.info(f"Creating ensemble for regime: {regime_name}")

            # Extract models and prepare data
            model_list = []
            model_names = []
            model_performances = {}

            for model_name, model_data in models.items():
                model_list.append(model_data["model"])
                model_names.append(model_name)

                # Calculate model performance on validation data
                if not validation_data.empty and "label" in validation_data.columns:
                    X_val = validation_data.drop("label", axis=1)
                    y_val = validation_data["label"]
                    
                    try:
                        predictions = model_data["model"].predict(X_val)
                        accuracy = accuracy_score(y_val, predictions)
                        precision = precision_score(y_val, predictions, average='weighted', zero_division=0)
                        recall = recall_score(y_val, predictions, average='weighted', zero_division=0)
                        f1 = f1_score(y_val, predictions, average='weighted', zero_division=0)
                        
                        model_performances[model_name] = {
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1
                        }
                    except Exception as e:
                        self.logger.warning(f"Could not evaluate model {model_name}: {e}")
                        model_performances[model_name] = {
                            "accuracy": 0.5,
                            "precision": 0.5,
                            "recall": 0.5,
                            "f1": 0.5
                        }

            # Create different ensemble types
            ensemble_results = {}

            # 1. StackingCV Ensemble with proper meta-learner
            stacking_ensemble = await self._create_stacking_ensemble(
                model_list,
                model_names,
                regime_name,
                training_data,
                validation_data,
            )
            ensemble_results["stacking_cv"] = stacking_ensemble

            # 2. Dynamic Performance-Weighted Ensemble using Sharpe Ratio
            dynamic_ensemble = await self._create_dynamic_weighting_ensemble(
                model_list,
                model_names,
                regime_name,
                training_data,
                validation_data,
                model_performances,
            )
            ensemble_results["dynamic_weighting"] = dynamic_ensemble

            # 3. Voting Ensemble
            voting_ensemble = await self._create_voting_ensemble(
                model_list,
                model_names,
                regime_name,
            )
            ensemble_results["voting"] = voting_ensemble

            return ensemble_results

        except Exception as e:
            self.logger.error(f"Error creating ensemble for regime {regime_name}: {e}")
            raise

    async def _create_stacking_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        regime_name: str,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
    ) -> dict[str, Any]:
        """Create enhanced StackingCV ensemble with proper meta-learner."""
        try:
            if training_data.empty or "label" not in training_data.columns:
                raise ValueError("Training data is empty or missing labels")

            X_train = training_data.drop("label", axis=1)
            y_train = training_data["label"]

            # Create estimators for stacking
            estimators = [
                (name, model) for name, model in zip(model_names, models, strict=False)
            ]

            # Use stratified k-fold for better validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # Create meta-learner with regularization
            meta_learner = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0,  # Regularization strength
                penalty='l2',
                solver='lbfgs'
            )

            # Create stacking classifier
            stacking_classifier = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=cv,
                stack_method="predict_proba",
                n_jobs=-1,
                passthrough=False,  # Don't pass original features to meta-learner
            )

            # Fit the stacking classifier
            stacking_classifier.fit(X_train, y_train)

            # Evaluate on validation data
            validation_metrics = {}
            if not validation_data.empty and "label" in validation_data.columns:
                X_val = validation_data.drop("label", axis=1)
                y_val = validation_data["label"]
                
                val_predictions = stacking_classifier.predict(X_val)
                val_probabilities = stacking_classifier.predict_proba(X_val)
                
                validation_metrics = {
                    "accuracy": accuracy_score(y_val, val_predictions),
                    "precision": precision_score(y_val, val_predictions, average='weighted', zero_division=0),
                    "recall": recall_score(y_val, val_predictions, average='weighted', zero_division=0),
                    "f1": f1_score(y_val, val_predictions, average='weighted', zero_division=0),
                }

            # Cross-validation scores
            cv_scores = cross_val_score(
                stacking_classifier, X_train, y_train, 
                cv=cv, scoring='accuracy', n_jobs=-1
            )

            ensemble_data = {
                "ensemble": stacking_classifier,
                "ensemble_type": "StackingCV",
                "base_models": model_names,
                "meta_learner": "LogisticRegression",
                "cv_folds": 5,
                "regime": regime_name,
                "creation_date": datetime.now().isoformat(),
                "validation_metrics": validation_metrics,
                "cv_scores": {
                    "mean": cv_scores.mean(),
                    "std": cv_scores.std(),
                    "scores": cv_scores.tolist()
                },
                "features": {
                    "use_probabilities": True,
                    "use_raw_predictions": False,
                    "passthrough": False,
                },
            }

            return ensemble_data

        except Exception as e:
            self.logger.error(
                f"Error creating stacking ensemble for {regime_name}: {e}",
            )
            raise

    async def _create_dynamic_weighting_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        regime_name: str,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        model_performances: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Create dynamic performance-weighted ensemble using Sharpe Ratio."""
        try:
            if validation_data.empty or "label" not in validation_data.columns:
                raise ValueError("Validation data is empty or missing labels")

            X_val = validation_data.drop("label", axis=1)
            y_val = validation_data["label"]

            # Calculate Sharpe Ratio for each model
            model_sharpe_ratios = {}
            model_weights = {}

            for i, (name, model) in enumerate(zip(model_names, models, strict=False)):
                try:
                    # Get predictions and probabilities
                    predictions = model.predict(X_val)
                    probabilities = model.predict_proba(X_val)
                    
                    # Calculate returns based on predictions (simplified approach)
                    # In a real implementation, you would calculate actual trading returns
                    correct_predictions = (predictions == y_val).astype(int)
                    
                    # Calculate "returns" based on prediction accuracy
                    # This is a simplified approach - in reality you'd calculate actual trading returns
                    returns = []
                    for j, (pred, actual) in enumerate(zip(predictions, y_val)):
                        if pred == actual:
                            # Correct prediction: positive return
                            returns.append(0.01)  # 1% return
                        else:
                            # Incorrect prediction: negative return
                            returns.append(-0.005)  # -0.5% return
                    
                    returns = np.array(returns)
                    
                    # Calculate Sharpe Ratio
                    if len(returns) > 0:
                        mean_return = np.mean(returns)
                        std_return = np.std(returns)
                        
                        if std_return > 0:
                            sharpe_ratio = mean_return / std_return
                        else:
                            sharpe_ratio = 0.0
                    else:
                        sharpe_ratio = 0.0
                    
                    model_sharpe_ratios[name] = sharpe_ratio
                    
                    # Models with negative Sharpe Ratio get weight 0
                    if sharpe_ratio <= 0:
                        model_weights[name] = 0.0
                    else:
                        model_weights[name] = sharpe_ratio
                        
                except Exception as e:
                    self.logger.warning(f"Could not calculate Sharpe Ratio for model {name}: {e}")
                    model_sharpe_ratios[name] = 0.0
                    model_weights[name] = 0.0

            # Normalize weights so they sum to 1
            total_weight = sum(model_weights.values())
            if total_weight > 0:
                for name in model_weights:
                    model_weights[name] /= total_weight
            else:
                # If all weights are 0, assign equal weights
                for name in model_weights:
                    model_weights[name] = 1.0 / len(model_weights)



            # Create ensemble object using the module-level class
            ensemble = DynamicWeightedEnsemble(models, model_names, model_weights)

            # Evaluate ensemble performance
            ensemble_predictions = ensemble.predict(X_val)
            ensemble_metrics = {
                "accuracy": accuracy_score(y_val, ensemble_predictions),
                "precision": precision_score(y_val, ensemble_predictions, average='weighted', zero_division=0),
                "recall": recall_score(y_val, ensemble_predictions, average='weighted', zero_division=0),
                "f1": f1_score(y_val, ensemble_predictions, average='weighted', zero_division=0),
            }

            ensemble_data = {
                "ensemble": ensemble,
                "ensemble_type": "DynamicWeightedEnsemble",
                "base_models": model_names,
                "weights": model_weights,
                "sharpe_ratios": model_sharpe_ratios,
                "regime": regime_name,
                "creation_date": datetime.now().isoformat(),
                "validation_metrics": ensemble_metrics,
                "features": {
                    "sharpe_ratio_weighting": True,
                    "negative_sharpe_filtering": True,
                    "dynamic_weighting": True,
                },
            }

            return ensemble_data

        except Exception as e:
            self.logger.error(
                f"Error creating dynamic weighting ensemble for {regime_name}: {e}",
            )
            raise

    async def _create_voting_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        regime_name: str,
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
                "regime": regime_name,
                "creation_date": datetime.now().isoformat(),
            }

            return ensemble_data

        except Exception as e:
            self.logger.error(f"Error creating voting ensemble for {regime_name}: {e}")
            raise


# For backward compatibility with existing step structure
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    **kwargs,
) -> bool:
    """
    Run the analyst ensemble creation step.

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
        step = AnalystEnsembleCreationStep(config)
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
        print(f"❌ Analyst ensemble creation failed: {e}")
        return False


if __name__ == "__main__":
    # Test the step
    async def test():
        result = await run_step("ETHUSDT", "BINANCE", "data/training")
        print(f"Test result: {result}")

    asyncio.run(test())
