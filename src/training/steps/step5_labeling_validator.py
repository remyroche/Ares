#!/usr/bin/env python3
"""Validator for Step 5: Labeling.

This module validates the labeling step outputs.
"""

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

logger = system_logger.getChild("Step5LabelingValidator")


class Step5LabelingValidator:
    """Validator for Step 5: Labeling."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = logger

    @validate_step2_operation
    def validate_step5_labeling(
        self, symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any]
    ) -> bool:
        """Validate Step 5: Labeling.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            training_input: Training input data

        Returns:
            bool: True if validation passes
        """
        self.logger.info("🔍 Starting Step 5: Labeling validation")

        try:
            # Check if labeled data directory exists
            labeled_data_dir = Path(data_dir) / "training" / "labeled_data"
            if not labeled_data_dir.exists():
                self.logger.warning(
                    f"⚠️ Labeled data directory not found: {labeled_data_dir}"
                )
                return False

            # Validate labeled data files
            labeled_files = list(labeled_data_dir.glob("*.parquet"))
            if not labeled_files:
                self.logger.warning("⚠️ No labeled data files found")
                return False

            # Validate each labeled file
            for labeled_file in labeled_files:
                if not self._validate_labeled_file(labeled_file):
                    return False

            # Check for labeling metadata file
            metadata_file = labeled_data_dir / f"{exchange}_{symbol}_1m_labeling_metadata.json"
            if not metadata_file.exists():
                self.logger.warning(f"⚠️ Labeling metadata file not found: {metadata_file}")
                return False

            # Validate metadata file
            if not self._validate_metadata_file(metadata_file):
                return False

            self.logger.info("✅ Step 5: Labeling validation passed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Step 5 validation failed: {e}")
            return False

    @validate_file_operation
    def _validate_labeled_file(self, labeled_file: Path) -> bool:
        """Validate a labeled data file."""
        try:
            self.logger.info(f"📁 Validating labeled file: {labeled_file.name}")

            # Load and validate the labeled file
            df = pd.read_parquet(labeled_file)

            # Check basic requirements
            if df.empty:
                self.logger.warning(f"⚠️ Labeled file is empty: {labeled_file.name}")
                return False

            # Check for required columns
            required_columns = ["timestamp", "label"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(
                    f"⚠️ Missing required columns in {labeled_file.name}: {missing_columns}"
                )
                return False

            # Check for feature columns (should have more than just timestamp and label)
            feature_columns = [col for col in df.columns if col not in required_columns]
            if len(feature_columns) < 5:  # Minimum number of features
                self.logger.warning(
                    f"⚠️ Insufficient features in {labeled_file.name}: {len(feature_columns)} features"
                )
                return False

            # Check label distribution
            label_counts = df["label"].value_counts()
            unique_labels = len(label_counts)
            if unique_labels < 2:
                self.logger.warning(
                    f"⚠️ Insufficient label diversity in {labeled_file.name}: {unique_labels} labels"
                )
                return False

            # Check for balanced labels (should have reasonable distribution)
            total_samples = len(df)
            min_samples_per_label = total_samples * 0.1  # At least 10% per label
            for label, count in label_counts.items():
                if count < min_samples_per_label:
                    self.logger.warning(
                        f"⚠️ Label {label} has too few samples in {labeled_file.name}: {count} (minimum {min_samples_per_label:.0f})"
                    )

            self.logger.info(
                f"✅ Labeled file validated: {len(df)} rows, {len(feature_columns)} features, {unique_labels} labels"
            )
            self.logger.info(f"   Label distribution: {label_counts.to_dict()}")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error validating labeled file {labeled_file}: {e}")
            return False

    @validate_file_operation
    def _validate_metadata_file(self, metadata_file: Path) -> bool:
        """Validate a labeling metadata file."""
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
            required_fields = ["labeling_method", "label_distribution", "total_samples"]
            missing_fields = [field for field in required_fields if field not in metadata]
            if missing_fields:
                self.logger.warning(
                    f"⚠️ Missing required fields in {metadata_file.name}: {missing_fields}"
                )
                return False

            # Validate total_samples
            total_samples = metadata.get("total_samples", 0)
            if total_samples < 100:  # Minimum data requirement
                self.logger.warning(
                    f"⚠️ Insufficient total samples in {metadata_file.name}: {total_samples}"
                )
                return False

            # Validate label distribution
            label_distribution = metadata.get("label_distribution", {})
            if not isinstance(label_distribution, dict):
                self.logger.warning(
                    f"⚠️ Invalid label distribution format in {metadata_file.name}"
                )
                return False

            if len(label_distribution) < 2:
                self.logger.warning(
                    f"⚠️ Insufficient label diversity in {metadata_file.name}: {len(label_distribution)} labels"
                )
                return False

            self.logger.info(f"✅ Metadata file validated: {total_samples} samples, {len(label_distribution)} labels")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error validating metadata file {metadata_file}: {e}")
            return False


@validate_step2_operation
def step5_labeling_validator(
    symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any], config: dict[str, Any]
) -> bool:
    """Step 5: Labeling Validator.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        training_input: Training input data
        config: Configuration dictionary

    Returns:
        bool: True if validation passes
    """
    logger.info("🔍 Starting Step 5: Labeling validation")

    try:
        validator = Step5LabelingValidator(config)
        result = validator.validate_step5_labeling(
            symbol, exchange, data_dir, training_input
        )

        if result:
            logger.info("✅ Step 5: Labeling validation passed")
            return True
        else:
            logger.warning("⚠️ Step 5: Labeling validation failed")
            return False

    except Exception as e:
        logger.exception(f"❌ Step 5 validation failed: {e}")
        return False