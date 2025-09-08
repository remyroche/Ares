# src/training/steps/step11_*.py

from pathlib import Path
from typing import Any

import joblib

from .core.decorators import validates
from .utils.common_operations import safe_json_load
from src.utils.logger import system_logger
import json
import logging
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler


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
            # Fast fail: Validate input parameters
            if not self._validate_input_parameters(symbol, exchange, data_dir, training_input):
                return False
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

    def _validate_input_parameters(self, symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any]) -> bool:
        """Validate input parameters with comprehensive checks."""
        try:
            # Validate symbol
            if not symbol or not isinstance(symbol, str) or len(symbol.strip()) == 0:
                self.logger.error("❌ Invalid symbol: must be non-empty string")
                return False
            
            # Validate exchange
            if not exchange or not isinstance(exchange, str) or len(exchange.strip()) == 0:
                self.logger.error("❌ Invalid exchange: must be non-empty string")
                return False
            
            # Validate data directory
            if not data_dir or not isinstance(data_dir, str):
                self.logger.error("❌ Invalid data_dir: must be non-empty string")
                return False
            
            data_path = Path(data_dir)
            if not data_path.exists():
                self.logger.error(f"❌ Data directory does not exist: {data_dir}")
                return False
            
            if not data_path.is_dir():
                self.logger.error(f"❌ Data directory is not a directory: {data_dir}")
                return False
            
            # Validate training input
            if not isinstance(training_input, dict):
                self.logger.error("❌ training_input must be a dictionary")
                return False
            
            # Check for required keys in training_input
            required_keys = ['regime_data', 'features', 'targets']
            missing_keys = [key for key in required_keys if key not in training_input]
            if missing_keys:
                self.logger.error(f"❌ Missing required keys in training_input: {missing_keys}")
                return False
            
            # Validate regime data structure
            regime_data = training_input.get('regime_data', {})
            if not isinstance(regime_data, dict) or not regime_data:
                self.logger.error("❌ regime_data must be a non-empty dictionary")
                return False
            
            # Validate each regime
            for regime_name, regime_info in regime_data.items():
                if not isinstance(regime_info, dict):
                    self.logger.error(f"❌ Invalid regime_info type for {regime_name}")
                    return False
                
                if 'features' not in regime_info or 'targets' not in regime_info:
                    self.logger.error(f"❌ Missing features or targets for regime {regime_name}")
                    return False
            
            self.logger.info("✅ Input parameter validation passed")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Error validating input parameters: {e}")
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
