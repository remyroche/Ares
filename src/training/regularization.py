from typing import Any

# Import necessary ensemble types for type hinting and applying regularization
# These imports are here to allow the apply_regularization_to_ensembles method
# to correctly apply the config to the ensemble instances.
from src.analyst.predictive_ensembles.ensemble_orchestrator import (
    RegimePredictiveEnsembles,
)
from src.analyst.predictive_ensembles.regime_ensembles.base_ensemble import BaseEnsemble

# Ensure these imports are correct relative to the project root
from src.config import CONFIG
from src.utils.logger import system_logger


class RegularizationManager:
    """
    Manages the L1-L2 regularization configuration for the Ares Trading Bot's
    machine learning models. It extracts, applies, and validates regularization
    parameters from the global configuration.
    """

    def __init__(self):
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
    ):
        """
        Applies the loaded L1-L2 regularization configuration to all ensemble instances.
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
    ):
        """Applies regularization configuration to a specific ensemble instance."""
        try:
            # Check if the ensemble instance has a 'regularization_config' attribute
            # and set it. This is how the ensemble models access the parameters.
            if hasattr(ensemble_instance, "regularization_config"):
                ensemble_instance.regularization_config = self.regularization_config
            else:
                # If not present, add it. This ensures it's available for model creation.
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
        """
        Validates regularization configuration and reports on the setup.

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

            # Report on each model type's regularization setup
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
            sklearn_config = self.regularization_config.get("sklearn", {})
            print(
                f"   - alpha (for Ridge/Lasso): {sklearn_config.get('alpha', 'Not set')}",
            )
            print(
                f"   - l1_ratio (for ElasticNet): {sklearn_config.get('l1_ratio', 'Not set')}",
            )
            print(
                f"   - C (for LogisticRegression): {sklearn_config.get('C', 'Not set')}",
            )

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
