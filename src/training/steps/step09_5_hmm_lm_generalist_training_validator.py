# src/training/steps/step9_5_hmm_lm_generalist_training_validator.py

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logger import system_logger
from src.utils.validation_decorators import (
    validate_file_operation,
    validate_dataframe_operation,
    validate_step2_operation,
)

logger = system_logger.getChild("Step9_5HMMLMGeneralistTrainingValidator")


class Step9_5HMMLMGeneralistTrainingValidator:
    """Validator for Step 9.5: HMM LM Generalist Training."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = logger

    @validate_step2_operation
    def validate_step9_5_hmm_lm_generalist_training(
        self, symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any]
    ) -> bool:
        """Validate Step 9.5: HMM LM Generalist Training.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            training_input: Training input data

        Returns:
            bool: True if validation passes
        """
        self.logger.info("🔍 Starting Step 9.5: HMM LM Generalist Training validation")

        try:
            # Check if HMM LM generalist models directory exists
            hmm_lm_models_dir = Path(data_dir) / "training" / "hmm_lm_generalist_models"
            if not hmm_lm_models_dir.exists():
                self.logger.warning(
                    f"⚠️ HMM LM generalist models directory not found: {hmm_lm_models_dir}"
                )
                return False

            # Validate model files
            model_files = list(hmm_lm_models_dir.glob("*.joblib"))
            if not model_files:
                self.logger.warning("⚠️ No HMM LM generalist model files found")
                return False

            # Validate each model file
            for model_file in model_files:
                if not self._validate_model_file(model_file):
                    return False

            # Check for training metadata file
            metadata_file = hmm_lm_models_dir / f"{exchange}_{symbol}_1m_hmm_lm_training_metadata.json"
            if not metadata_file.exists():
                self.logger.warning(f"⚠️ HMM LM training metadata file not found: {metadata_file}")
                return False

            # Validate metadata file
            if not self._validate_metadata_file(metadata_file):
                return False

            self.logger.info("✅ Step 9.5: HMM LM Generalist Training validation passed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Step 9.5 validation failed: {e}")
            return False

    @validate_file_operation
    def _validate_model_file(self, model_file: Path) -> bool:
        """Validate an HMM LM generalist model file."""
        try:
            self.logger.info(f"📁 Validating HMM LM model: {model_file.name}")

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
                
                self.logger.info(f"✅ HMM LM model file validated: {model_file.name} ({file_size} bytes)")
                return True
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load model file {model_file.name}: {e}")
                return False

        except Exception as e:
            self.logger.exception(f"❌ Error validating model file {model_file}: {e}")
            return False

    @validate_file_operation
    def _validate_metadata_file(self, metadata_file: Path) -> bool:
        """Validate an HMM LM training metadata file."""
        try:
            self.logger.info(f"📁 Validating metadata file: {metadata_file.name}")

            # Load and validate the metadata file
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Check if metadata is a dictionary
            if not isinstance(metadata, dict):
                self.logger.warning(f"⚠️ Metadata file is not a valid JSON object: {metadata_file.name}")
                return False

            # Check for required fields
            required_fields = ["training_date", "model_count", "training_metrics"]
            missing_fields = [field for field in required_fields if field not in metadata]
            if missing_fields:
                self.logger.warning(
                    f"⚠️ Missing required fields in {metadata_file.name}: {missing_fields}"
                )
                return False

            # Validate model count
            model_count = metadata.get("model_count", 0)
            if model_count < 1:
                self.logger.warning(
                    f"⚠️ Invalid model count in {metadata_file.name}: {model_count}"
                )
                return False

            # Validate training metrics
            training_metrics = metadata.get("training_metrics", {})
            if not isinstance(training_metrics, dict):
                self.logger.warning(
                    f"⚠️ Invalid training metrics format in {metadata_file.name}"
                )
                return False

            # Check for basic metrics
            basic_metrics = ["accuracy", "loss", "training_time"]
            for metric in basic_metrics:
                if metric in training_metrics:
                    value = training_metrics[metric]
                    if isinstance(value, (int, float)) and value < 0:
                        self.logger.warning(
                            f"⚠️ Invalid {metric} value in {metadata_file.name}: {value}"
                        )

            self.logger.info(f"✅ Metadata file validated: {model_count} models, {len(training_metrics)} metrics")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error validating metadata file {metadata_file}: {e}")
            return False


@validate_step2_operation
def step9_5_hmm_lm_generalist_training_validator(
    symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any], config: dict[str, Any]
) -> bool:
    """Step 9.5: HMM LM Generalist Training Validator.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        training_input: Training input data
        config: Configuration dictionary

    Returns:
        bool: True if validation passes
    """
    logger.info("🔍 Starting Step 9.5: HMM LM Generalist Training validation")

    try:
        validator = Step9_5HMMLMGeneralistTrainingValidator(config)
        result = validator.validate_step9_5_hmm_lm_generalist_training(
            symbol, exchange, data_dir, training_input
        )

        if result:
            logger.info("✅ Step 9.5: HMM LM Generalist Training validation passed")
            return True
        else:
            logger.warning("⚠️ Step 9.5: HMM LM Generalist Training validation failed")
            return False

    except Exception as e:
        logger.exception(f"❌ Step 9.5 validation failed: {e}")
        return False