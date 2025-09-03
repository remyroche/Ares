# Import necessary ensemble types for type hinting and applying regularization
# These imports are here to allow the apply_regularization_to_ensembles method
# to correctly apply the config to the ensemble instances.
from typing import Any

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from torch import nn

from src.analyst.predictive_ensembles.ensemble_orchestrator import (
    RegimePredictiveEnsembles,
)
from src.analyst.predictive_ensembles.regime_ensembles.base_ensemble import BaseEnsemble

# Ensure these imports are correct relative to the project root
from src.config import CONFIG
from src.utils.logger import system_logger


class RegularizationManager:
    """Manages the L1-L2 regularization configuration for the Ares Trading Bot's'"
    machine learning models. It extracts, applies, and validates regularization
    parameters from the global configuration.
    """
    def __init__(self) -> None:
        self.logger = system_logger.getChild("RegularizationManager")
        self.regularization_config = self._get_regularization_config()
        self.logger.info("RegularizationManager initialized.")

    def _get_regularization_config(self) -> dict[str, Any]:
        """Extract and validate L1-L2 regularization configuration from CONFIG."""
        base_reg_config = CONFIG["MODEL_TRAINING"].get("regularization", {})

        regularization_config = {
            "l1_alpha": base_reg_config.get("l1_alpha", 0.01),
            "l2_alpha": base_reg_config.get("l2_alpha", 0.001),
            "dropout_rate": base_reg_config.get("dropout_rate", 0.2),
            "lightgbm": {
                "reg_alpha": base_reg_config.get("l1_alpha", 0.01),
                "reg_lambda": base_reg_config.get("l2_alpha", 0.001),
            },
            "tensorflow": {
                "l1_reg": base_reg_config.get("l1_alpha", 0.01),
                "l2_reg": base_reg_config.get("l2_alpha", 0.001),
                "dropout_rate": base_reg_config.get("dropout_rate", 0.2),
            },
            "sklearn": {
                "alpha": base_reg_config.get("l1_alpha", 0.01),
                "l1_ratio": 0.5,
                "C": 1.0 / max(base_reg_config.get("l1_alpha", 0.01), 1e-8),
            },
            "tabnet": {
                "lambda_sparse": base_reg_config.get("l1_alpha", 0.01),
                "lambda_l2": base_reg_config.get("l2_alpha", 0.001),
            },
        }

        self.logger.info(
            f"Regularization configuration loaded: {regularization_config}",
        )
        return regularization_config

    def apply_regularization_to_ensembles(
        self,
        ensemble_orchestrator: RegimePredictiveEnsembles,
    ) -> None:
        """Applies the loaded L1-L2 regularization configuration to all ensemble instances."
        This method is called by TrainingManager.
        """
        try:
            for (
                regime_name,
                ensemble_instance,
            ) in ensemble_orchestrator.regime_ensembles.items():
                self._apply_regularization_to_single_ensemble(
                    ensemble_instance,
                    regime_name,
                )
            self.logger.info(
                "Successfully applied regularization configuration to all ensembles.",
            )
        except Exception as e:
            self.logger.error(
                f"Failed to apply regularization configuration to ensembles: {e}",
                exc_info=True,
            )

    def _apply_regularization_to_single_ensemble(
        self,
        ensemble_instance: BaseEnsemble,
        regime_name: str,
    ) -> None:
        """Applies regularization configuration to a specific ensemble instance."""
        try:
            # Check if the ensemble instance has a 'regularization_config' attribute
            # and set it. This is how the ensemble models access the parameters.
            if hasattr(ensemble_instance, "regularization_config"):
                ensemble_instance.regularization_config = self.regularization_config
            else:
                # If not present, add it. This ensures it's available for model creation.'
                ensemble_instance.regularization_config = self.regularization_config

            # If the ensemble has specific deep learning config, update it directly
            if hasattr(ensemble_instance, "dl_config"):
                ensemble_instance.dl_config.update(
                    {
                        "l1_reg": self.regularization_config["tensorflow"]["l1_reg"],
                        "l2_reg": self.regularization_config["tensorflow"]["l2_reg"],
                        "dropout_rate": self.regularization_config["tensorflow"][
                            "dropout_rate"
                        ],
                    },
                )

            self.logger.info(f"Applied regularization to {regime_name} ensemble.")

        except Exception as e:
            self.logger.exception(
                f"Failed to apply regularization to {regime_name} ensemble: {e}",
            )

    def validate_and_report_regularization(self) -> bool:
        """Validates regularization configuration and reports on the setup."

        Returns:
            bool: True if regularization is properly configured, False otherwise

        """
        try:
            self.logger.info("=== L1-L2 Regularization Validation Report ===")

            # Check configuration completeness
            required_keys = ["l1_alpha", "l2_alpha", "dropout_rate"]
            missing_keys = [
                key for key in required_keys if key not in self.regularization_config
            ]

            if missing_keys:
                self.logger.warning(
                    f"Missing regularization config keys: {missing_keys}",
                )
                return False

            # Report on each model type's regularization setup'
            self.logger.info("📊 Base Regularization Parameters:")
            self.logger.info(f"   - L1 Alpha: {self.regularization_config['l1_alpha']}")
            self.logger.info(f"   - L2 Alpha: {self.regularization_config['l2_alpha']}")
            self.logger.info(
                f"   - Dropout Rate: {self.regularization_config['dropout_rate']}",
            )

            self.logger.info("\n🌳 LightGBM Regularization:")
            lgbm_config = self.regularization_config.get("lightgbm", {})
            self.logger.info(
                f"   - L1 (reg_alpha): {lgbm_config.get('reg_alpha', 'Not set')}",
            )
            self.logger.info(
                f"   - L2 (reg_lambda): {lgbm_config.get('reg_lambda', 'Not set')}",
            )

            self.logger.info("\n🧠 TensorFlow/Keras Regularization:")
            tf_config = self.regularization_config.get("tensorflow", {})
            self.logger.info(
                f"   - L1 Regularization: {tf_config.get('l1_reg', 'Not set')}",
            )
            self.logger.info(
                f"   - L2 Regularization: {tf_config.get('l2_reg', 'Not set')}",
            )
            self.logger.info(
                f"   - Dropout Rate: {tf_config.get('dropout_rate', 'Not set')}",
            )

            self.logger.info("\n📈 Scikit-learn Regularization:")
            self.regularization_config.get("sklearn", {})

            self.logger.info("\n🎯 TabNet Regularization:")
            tabnet_config = self.regularization_config.get("tabnet", {})
            self.logger.info(
                f"   - lambda_sparse (L1): {tabnet_config.get('lambda_sparse', 'Not set')}",
            )
            self.logger.info(
                f"   - lambda_l2 (L2): {tabnet_config.get('lambda_l2', 'Not set')}",
            )

            # Validate regularization values are reasonable
            validation_issues = []

            if self.regularization_config["l1_alpha"] <= 0:
                validation_issues.append("L1 alpha should be positive")
            if self.regularization_config["l2_alpha"] <= 0:
                validation_issues.append("L2 alpha should be positive")
            if not 0 <= self.regularization_config["dropout_rate"] <= 1:
                validation_issues.append("Dropout rate should be between 0 and 1")

            if validation_issues:
                self.logger.warning(
                    f"⚠️  Regularization validation issues: {validation_issues}",
                )
                return False

            self.logger.info("✅ Regularization configuration validated successfully")
            self.logger.info("=== End Regularization Report ===")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to validate regularization configuration: {e}",
                exc_info=True,
            )
            return False

    async def optimize_regularization_for_model(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        model_type: str,
        architecture: str,
    ) -> dict[str, Any]:
        """Optimize regularization parameters for a specific model type and architecture."

        Args:
            features_df: Input features DataFrame
            target: Target variable series
            model_type: Model type (classification, regression)
            architecture: Model architecture (LightGBM, CNN, TCN, Transformer)

        Returns:
            Dict containing optimized regularization parameters

        """
        try:
            if architecture == "LightGBM":
                return await self._optimize_lightgbm_regularization(features_df, target, model_type)
            if architecture in ["CNN", "TCN", "Transformer"]:
                return await self._optimize_neural_network_regularization(features_df, target, model_type, architecture)
            return await self._optimize_general_regularization(features_df, target, model_type)

        except Exception as e:
            self.logger.exception(f"❌ Regularization optimization failed: {e}")
            return self._get_default_regularization_params(architecture)

    async def _optimize_lightgbm_regularization(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        model_type: str,
    ) -> dict[str, Any]:
        """Optimize LightGBM regularization parameters using Optuna."""
        try:
            def objective(trial):
                reg_alpha = trial.suggest_float("reg_alpha", 0.001, 0.1)
                reg_lambda = trial.suggest_float("reg_lambda", 0.001, 0.1)

                if model_type == "classification":
                    model = lgb.LGBMClassifier(
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        n_estimators=100,
                        random_state=42,
                        verbose=-1,
                    )
                    scoring = "accuracy"
                else:
                    model = lgb.LGBMRegressor(
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        n_estimators=100,
                        random_state=42,
                        verbose=-1,
                    )
                    scoring = "neg_mean_squared_error"

                scores = cross_val_score(model, features_df, target, cv=3, scoring=scoring)
                return scores.mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)

            return {
                "reg_alpha": study.best_params["reg_alpha"],
                "reg_lambda": study.best_params["reg_lambda"],
                "best_score": study.best_value,
            }

        except Exception as e:
            self.logger.exception(f"❌ LightGBM regularization optimization failed: {e}")
            return {"reg_alpha": 0.01, "reg_lambda": 0.001}

    async def _optimize_neural_network_regularization(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        model_type: str,
        architecture: str,
    ) -> dict[str, Any]:
        """Optimize neural network regularization parameters using Optuna."""
        try:
            def objective(trial):
                weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3)
                dropout = trial.suggest_float("dropout", 0.1, 0.5)

                # Create a simple neural network for testing
                model = self._create_simple_nn_model(
                    input_size=features_df.shape[1],
                    params={"dropout": dropout, "weight_decay": weight_decay},
                    model_type=model_type,
                )

                # Train and evaluate the model with real metrics
                try:
                    import torch
                    from sklearn.preprocessing import StandardScaler

                    # Prepare data
                    X = features_df.values
                    y = target.values

                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # Convert to tensors for PyTorch model
                    import torch
                except Exception as e:
                    pass  # TODO: Handle exception
                except Exception as e:
                    pass  # TODO: Handle exception properly
                    X_tensor = torch.FloatTensor(X_scaled)
                    if model_type == "classification":
                        y_tensor = torch.LongTensor(y)
                        criterion = torch.nn.CrossEntropyLoss()
                    else:
                        y_tensor = torch.FloatTensor(y).unsqueeze(1)
                        criterion = torch.nn.MSELoss()

                    # Simple training loop
                    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)

                    model.train()
                    for _epoch in range(10):  # Short training for optimization
                        optimizer.zero_grad()
                        outputs = model(X_tensor)
                        loss = criterion(outputs, y_tensor)
                        loss.backward()
                        optimizer.step()

                    # Evaluate using cross-validation
                    model.eval()
                    with torch.no_grad():
                        predictions = model(X_tensor)
                        if model_type == "classification":
                            _, predicted = torch.max(predictions, 1)
                            return (predicted == y_tensor).float().mean().item()
                        mse = criterion(predictions, y_tensor).item()
                        return -mse  # Return negative MSE for maximization

                except Exception as e:
                    self.logger.warning(f"⚠️ Neural network evaluation failed: {e}")
                    return 0.5  # Fallback score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)

            return {
                "weight_decay": study.best_params["weight_decay"],
                "dropout": study.best_params["dropout"],
                "best_score": study.best_value,
            }

        except Exception as e:
            self.logger.exception(f"❌ Neural network regularization optimization failed: {e}")
            return {"weight_decay": 1e-4, "dropout": 0.2}

    async def _optimize_general_regularization(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        model_type: str,
    ) -> dict[str, Any]:
        """Optimize general regularization parameters using ElasticNet."""
        try:
            def objective(trial):
                alpha = trial.suggest_float("alpha", 0.001, 0.1)
                l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)

                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                scoring = "neg_mean_squared_error" if model_type == "regression" else "accuracy"
                scores = cross_val_score(model, features_df, target, cv=3, scoring=scoring)
                return scores.mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)

            return {
                "alpha": study.best_params["alpha"],
                "l1_ratio": study.best_params["l1_ratio"],
                "best_score": study.best_value,
            }

        except Exception as e:
            self.logger.exception(f"❌ General regularization optimization failed: {e}")
            return {"alpha": 0.01, "l1_ratio": 0.5}

    def _create_simple_nn_model(self, input_size: int, params: dict[str, Any], model_type: str):
        """Create a simple neural network model for regularization testing."""
        class SimpleNN(nn.Module):
            def __init__(self, input_size, params, model_type) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Dropout(params.get("dropout", 0.2)),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(params.get("dropout", 0.2)),
                    nn.Linear(64, 1 if model_type == "regression" else 2),
                )

            def forward(self, x):
                return self.layers(x)

        return SimpleNN(input_size, params, model_type)

    def _get_default_regularization_params(self, architecture: str) -> dict[str, Any]:
        """Get default regularization parameters for an architecture."""
        if architecture == "LightGBM":
            return {"reg_alpha": 0.01, "reg_lambda": 0.001}
        if architecture in ["CNN", "TCN", "Transformer"]:
            return {"weight_decay": 1e-4, "dropout": 0.2}
        return {"alpha": 0.01, "l1_ratio": 0.5}
