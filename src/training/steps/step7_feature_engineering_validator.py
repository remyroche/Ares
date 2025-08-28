# src/training/steps/step7_feature_engineering_validator.py

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

logger = system_logger.getChild("Step7FeatureEngineeringValidator")


class Step7FeatureEngineeringValidator:
    """Validator for Step 7: Advanced Feature Engineering (After Regime Discovery)."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = logger

    @validate_step2_operation
    def validate_step7_feature_engineering(
        self, symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any]
    ) -> bool:
        """Validate Step 7: Advanced Feature Engineering.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            training_input: Training input data

        Returns:
            bool: True if validation passes
        """
        self.logger.info("🔍 Starting Step 7: Advanced Feature Engineering validation")

        try:
            # Check if regime-aware features exist
            regime_features_dir = Path(data_dir) / "training" / "regime_features"
            if not regime_features_dir.exists():
                self.logger.warning(
                    f"⚠️ Regime features directory not found: {regime_features_dir}"
                )
                return False

            # Validate regime-specific feature files
            regime_dirs = [d for d in regime_features_dir.iterdir() if d.is_dir()]
            if not regime_dirs:
                self.logger.warning("⚠️ No regime-specific feature directories found")
                return False

            # Validate each regime's features
            for regime_dir in regime_dirs:
                regime_name = regime_dir.name
                self.logger.info(f"📊 Validating features for regime: {regime_name}")

                # Check for feature files
                feature_files = list(regime_dir.glob("*.parquet"))
                if not feature_files:
                    self.logger.warning(
                        f"⚠️ No feature files found for regime: {regime_name}"
                    )
                    continue

                # Validate feature file
                for feature_file in feature_files:
                    if not self._validate_feature_file(feature_file, regime_name):
                        return False

            self.logger.info("✅ Step 7: Advanced Feature Engineering validation passed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Step 7 validation failed: {e}")
            return False

    @validate_file_operation
    def _validate_feature_file(self, feature_file: Path, regime_name: str) -> bool:
        """Validate a feature file for a specific regime."""
        try:
            self.logger.info(f"📁 Validating feature file: {feature_file.name}")

            # Load and validate the feature file
            df = pd.read_parquet(feature_file)

            # Check basic requirements
            if df.empty:
                self.logger.warning(f"⚠️ Feature file is empty: {feature_file.name}")
                return False

            # Check for required columns
            required_columns = ["timestamp", "label"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(
                    f"⚠️ Missing required columns in {feature_file.name}: {missing_columns}"
                )
                return False

            # Check for feature columns (should have more than just timestamp and label)
            feature_columns = [col for col in df.columns if col not in required_columns]
            if len(feature_columns) < 5:  # Minimum number of features
                self.logger.warning(
                    f"⚠️ Insufficient features in {feature_file.name}: {len(feature_columns)} features"
                )
                return False

            # Check for regime-specific features
            regime_specific_features = [
                col for col in feature_columns if regime_name.lower() in col.lower()
            ]
            if not regime_specific_features:
                self.logger.info(
                    f"📊 No regime-specific features found for {regime_name} (this is acceptable)"
                )

            self.logger.info(
                f"✅ Feature file validated: {len(df)} rows, {len(feature_columns)} features"
            )
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error validating feature file {feature_file}: {e}")
            return False


@validate_step2_operation
def step7_feature_engineering_validator(
    symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any], config: dict[str, Any]
) -> bool:
    """Step 7: Advanced Feature Engineering Validator.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        training_input: Training input data
        config: Configuration dictionary

    Returns:
        bool: True if validation passes
    """
    logger.info("🔍 Starting Step 7: Advanced Feature Engineering validation")

    try:
        validator = Step7FeatureEngineeringValidator(config)
        result = validator.validate_step7_feature_engineering(
            symbol, exchange, data_dir, training_input
        )

        if result:
            logger.info("✅ Step 7: Advanced Feature Engineering validation passed")
            return True
        else:
            logger.warning("⚠️ Step 7: Advanced Feature Engineering validation failed")
            return False

    except Exception as e:
        logger.exception(f"❌ Step 7 validation failed: {e}")
        return False