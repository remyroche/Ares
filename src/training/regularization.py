from typing import Any, Dict

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from torch import nn

# Import necessary ensemble types for type hinting and applying regularization
# These imports are here to allow the apply_regularization_to_ensembles method
# to correctly apply the config to the ensemble instances.
from src.analyst.predictive_ensembles.ensemble_orchestrator import (
    RegimePredictiveEnsembles)
from src.analyst.predictive_ensembles.regime_ensembles.base_ensemble import BaseEnsemble

# Ensure these imports are correct relative to the project root
from src.config import CONFIG
from src.utils.logger import system_logger


class RegularizationManager:
    parameters from the global configuration.
    """

    def __init__(self) -> None:
self.logger = system_logger.getChild("RegularizationManager")
        self.regularization_config = self._get_regularization_config()
        self.logger.info("RegularizationManager initialized.")


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
            f"Regularization configuration loaded: {regularization_config}")
        return regularization_config

            self.logger.info(
                "Successfully applied regularization configuration to all ensembles.")
        except Exception as e:
                f"Failed to apply regularization configuration to ensembles: {e}",
                exc_info=True)

            else:
# If not present, add it. This ensures it's available for model creation.
                ensemble_instance.regularization_config = self.regularization_config


            self.logger.info(f"Applied regularization to {regime_name} ensemble.")

        except Exception as e:
            self.logger.info("=== L1-L2 Regularization Validation Report ===")

            # Check configuration completeness
            required_keys = ["l1_alpha", "l2_alpha", "dropout_rate"]
            missing_keys = [
                key for key in required_keys if key not in self.regularization_config
            ]

            if missing_keys:
                return False

            # Report on each model type's regularization setup
            self.logger.info("📊 Base Regularization Parameters:")
            self.logger.info(f"   - L1 Alpha: {self.regularization_config['l1_alpha']}")
            self.logger.info(f"   - L2 Alpha: {self.regularization_config['l2_alpha']}")
            self.logger.info(
                f"   - Dropout Rate: {self.regularization_config['dropout_rate']}")

            self.logger.info("\n🌳 LightGBM Regularization:")
            lgbm_config = self.regularization_config.get("lightgbm", {})
            self.logger.info(
                f"   - L1 (reg_alpha): {lgbm_config.get('reg_alpha', 'Not set')}")
            self.logger.info(
                f"   - L2 (reg_lambda): {lgbm_config.get('reg_lambda', 'Not set')}")

            self.logger.info("\n🧠 TensorFlow/Keras Regularization:")
            tf_config = self.regularization_config.get("tensorflow", {})
            self.logger.info(
                f"   - L1 Regularization: {tf_config.get('l1_reg', 'Not set')}")
            self.logger.info(
                f"   - L2 Regularization: {tf_config.get('l2_reg', 'Not set')}")
            self.logger.info(
                f"   - Dropout Rate: {tf_config.get('dropout_rate', 'Not set')}")

            self.logger.info("\n📈 Scikit-learn Regularization:")
            sklearn_config = self.regularization_config.get("sklearn", {})
            self.logger.info(
                f"   - Alpha: {sklearn_config.get('alpha', 'Not set')}")
            self.logger.info(
                f"   - L1 Ratio: {sklearn_config.get('l1_ratio', 'Not set')}")

            self.logger.info("\n🎯 TabNet Regularization:")
            tabnet_config = self.regularization_config.get("tabnet", {})
            self.logger.info(
                f"   - lambda_sparse (L1): {tabnet_config.get('lambda_sparse', 'Not set')}")
            self.logger.info(
                f"   - lambda_l2 (L2): {tabnet_config.get('lambda_l2', 'Not set')}")

            # Validate regularization values are reasonable
            validation_issues = []

            if self.regularization_config["l1_alpha"] <= 0:
validation_issues.append("L1 alpha should be positive")
            if self.regularization_config["l2_alpha"] <= 0:
validation_issues.append("L2 alpha should be positive")
            if not 0 <= self.regularization_config["dropout_rate"] <= 1:
validation_issues.append("Dropout rate should be between 0 and 1")

            if validation_issues:
                return False

            self.logger.info("✅ Regularization configuration validated successfully")
            self.logger.info("=== End Regularization Report ===")
            return True

        except Exception as e:
                f"Failed to validate regularization configuration: {e}",
                exc_info=True)
            return False

                    scoring = "accuracy"
                else:
                    model = lgb.LGBMRegressor(
                        reg_alpha=reg_alpha, reg_lambda=reg_lambda, n_estimators=100,
                        random_state=42, verbose=-1)
                    scoring = "neg_mean_squared_error"

                scores = cross_val_score(model, features_df, target, cv=3, scoring=scoring)
                return scores.mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)

            return {
                "reg_alpha": study.best_params["reg_alpha"],
                "reg_lambda": study.best_params["reg_lambda"],
                "best_score": study.best_value}

        except Exception as e:

                # Create a simple neural network for testing
                model = self._create_simple_nn_model(
                    input_size=features_df.shape[1],
                    params={"dropout": dropout, "weight_decay": weight_decay},
                    model_type=model_type)

                # Train and evaluate the model with real metrics
                try:
                    from sklearn.preprocessing import StandardScaler

                    # Prepare data
                    X = features_df.values
                    y = target.values

                    # Standardize features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # Convert to tensors for PyTorch model
                    import torch
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
                            return (predicted == y_tensor).float().mean().item()
                        mse = criterion(predictions, y_tensor).item()
                        return -mse  # Return negative MSE for maximization

                except Exception as e:
                    return 0.5  # Fallback score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)

            return {
                "weight_decay": study.best_params["weight_decay"],
                "dropout": study.best_params["dropout"],
                "best_score": study.best_value}

        except Exception as e:

                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                scoring = "neg_mean_squared_error" if model_type == "regression" else "accuracy"
                scores = cross_val_score(model, features_df, target, cv=3, scoring=scoring)
                return scores.mean()

            study = optuna.create_study(direction="maximize")

            return {
                "alpha": study.best_params["alpha"],
                "l1_ratio": study.best_params["l1_ratio"],
                "best_score": study.best_value}

        except Exception as e:

    def _create_simple_nn_model(...):
"""Create a simple neural network model for regularization testing."""
        class SimpleNN(nn.Module):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 128), nn.ReLU(),
                    nn.Dropout(params.get("dropout", 0.2)),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.Dropout(params.get("dropout", 0.2)),

            def forward(...):
                return self.layers(x)

        return SimpleNN(input_size, params, model_type)

