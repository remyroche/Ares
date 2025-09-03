# src/training/steps/step11_*.py

from src.core.domain import (

from src.core.decorators import validates
    validate_dataframe_operation,
    validate_file_operation,
    validate_step2_operation
)
from pathlib import Path
from typing import Any
from src.utils.common_operations import safe_json_load
from src.utils.logger import system_logger

logger = system_logger.getChild("Step11AnalystCreationValidator")


class Step11AnalystCreationValidator:
    """Validator for Step 11: Analyst Creation."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = logger

    @validates()
    def validate_step11_analyst_creation(
        self, symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any],
    ) -> bool:
        """Validate Step 11: Analyst Creation.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            training_input: Training input data

        Returns:
            bool: True if validation passes
        """
        self.logger.info("🔍 Starting Step 11: Analyst Creation validation")

        try:
            # Check if analyst models directory exists
            analyst_models_dir = Path(data_dir) / "analyst_models"
            if not analyst_models_dir.exists():
                self.logger.warning(
                    f"⚠️ Analyst models directory not found: {analyst_models_dir}",
                )
                return False

            # Validate regime-specific analyst model directories
            regime_dirs = [d for d in analyst_models_dir.iterdir() if d.is_dir()]
            if not regime_dirs:
                self.logger.warning("⚠️ No regime-specific analyst model directories found")
                return False

            # Validate each regime's analyst models
            for regime_dir in regime_dirs:
                regime_name = regime_dir.name
                self.logger.info(f"📊 Validating analyst models for regime: {regime_name}")

                # Check for model files
                model_files = list(regime_dir.glob("*.joblib"))
                if not model_files:
                    self.logger.warning(
                        f"⚠️ No analyst model files found for regime: {regime_name}",
                    )
                    continue

                # Validate each model file
                for model_file in model_files:
                    if not self._validate_analyst_model(model_file, regime_name):
                        return False

                # Check for metadata files
                metadata_files = list(regime_dir.glob("*_metadata.json"))
                if not metadata_files:
                    self.logger.warning(
                        f"⚠️ No metadata files found for regime: {regime_name}",
                    )
                    continue

                # Validate metadata files
                for metadata_file in metadata_files:
                    if not self._validate_metadata_file(metadata_file, regime_name):
                        return False

            self.logger.info("✅ Step 11: Analyst Creation validation passed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Step 11 validation failed: {e}")
            return False

    @validates()
    def _validate_analyst_model(self, model_file: Path, regime_name: str) -> bool:
        """Validate an analyst model file."""
        try:
            self.logger.info(f"📁 Validating analyst model: {model_file.name}")

            # Check file size (should be reasonable for a model)
            file_size = model_file.stat().st_size
            if file_size < 1000:  # Less than 1KB is suspicious
                self.logger.warning(f"⚠️ Model file seems too small: {file_size} bytes")
                return False

            # Try to load the model to ensure it's valid
            try:
                import joblib
                model = joblib.load(model_file)
                if model is None:
                    self.logger.warning(f"⚠️ Model file is empty: {model_file.name}")
                    return False

                self.logger.info(f"✅ Model file validated: {model_file.name} ({file_size} bytes)")
                return True
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load model file {model_file.name}: {e}")
                return False

        except Exception as e:
            self.logger.exception(f"❌ Error validating model file {model_file}: {e}")
            return False

    @validates()
    def _validate_metadata_file(self, metadata_file: Path, regime_name: str) -> bool:
        """Validate a metadata file."""
        try:
            self.logger.info(f"📁 Validating metadata file: {metadata_file.name}")

            # Load and validate the metadata file
            metadata = safe_json_load(metadata_file)

            # Check required fields
            required_fields = ["accuracy", "model_type", "creation_date"]
            missing_fields = [field for field in required_fields if field not in metadata]
            if missing_fields:
                self.logger.warning(
                    f"⚠️ Missing required fields in {metadata_file.name}: {missing_fields}",
                )
                return False

            # Validate accuracy (should be between 0 and 1)
            accuracy = metadata.get("accuracy", 0.0)
            if not (0.0 <= accuracy <= 1.0):
                self.logger.warning(
                    f"⚠️ Invalid accuracy value in {metadata_file.name}: {accuracy}",
                )
                return False

            # Validate model type
            model_type = metadata.get("model_type", "")
            valid_types = ["lightgbm", "xgboost", "random_forest", "neural_network"]
            if model_type not in valid_types:
                self.logger.warning(
                    f"⚠️ Invalid model type in {metadata_file.name}: {model_type}",
                )
                return False

            self.logger.info(
                f"✅ Metadata file validated: {metadata_file.name} (accuracy: {accuracy:.4f}, type: {model_type})",
            )
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error validating metadata file {metadata_file}: {e}")
            return False


@validates()
def step11_analyst_creation_validator(
    symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any], config: dict[str, Any],
) -> bool:
    """Step 11: Analyst Creation Validator.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        training_input: Training input data
        config: Configuration dictionary

    Returns:
        bool: True if validation passes
    """
    logger.info("🔍 Starting Step 11: Analyst Creation validation")

    try:
        validator = Step11AnalystCreationValidator(config)
        result = validator.validate_step11_analyst_creation(
            symbol, exchange, data_dir, training_input,
        )

        if result:
            logger.info("✅ Step 11: Analyst Creation validation passed")
            return True
        logger.warning("⚠️ Step 11: Analyst Creation validation failed")
        return False

    except Exception as e:
        logger.exception(f"❌ Step 11 validation failed: {e}")
        return False
