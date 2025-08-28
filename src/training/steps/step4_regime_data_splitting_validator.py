#!/usr/bin/env python3
"""Validator for Step 4: Regime Data Splitting.

This module validates the regime data splitting step outputs with support for 10+ regimes.
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

logger = system_logger.getChild("Step4RegimeDataSplittingValidator")


class Step4RegimeDataSplittingValidator:
    """Validator for Step 4: Regime Data Splitting."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = logger

    @validate_step2_operation
    def validate_step4_regime_data_splitting(
        self, symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any]
    ) -> bool:
        """Validate Step 4: Regime Data Splitting.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            training_input: Training input data

        Returns:
            bool: True if validation passes
        """
        self.logger.info("🔍 Starting Step 4: Regime Data Splitting validation")

        try:
            # Check if regime data splitting directory exists
            regime_splits_dir = Path(data_dir) / "training" / "regime_splits"
            if not regime_splits_dir.exists():
                self.logger.warning(
                    f"⚠️ Regime splits directory not found: {regime_splits_dir}"
                )
                return False

            # Validate regime split files
            regime_files = list(regime_splits_dir.glob("*.parquet"))
            if not regime_files:
                self.logger.warning("⚠️ No regime split files found")
                return False

            # Validate each regime file
            for regime_file in regime_files:
                if not self._validate_regime_file(regime_file):
                    return False

            # Check for regime statistics file
            stats_file = regime_splits_dir / f"{exchange}_{symbol}_1m_regime_statistics.json"
            if not stats_file.exists():
                self.logger.warning(f"⚠️ Regime statistics file not found: {stats_file}")
                return False

            # Validate statistics file
            if not self._validate_statistics_file(stats_file):
                return False

            self.logger.info("✅ Step 4: Regime Data Splitting validation passed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Step 4 validation failed: {e}")
            return False

    @validate_file_operation
    def _validate_regime_file(self, regime_file: Path) -> bool:
        """Validate a regime split file."""
        try:
            self.logger.info(f"📁 Validating regime file: {regime_file.name}")

            # Load and validate the regime file
            df = pd.read_parquet(regime_file)

            # Check basic requirements
            if df.empty:
                self.logger.warning(f"⚠️ Regime file is empty: {regime_file.name}")
                return False

            # Check for required columns
            required_columns = ["timestamp", "composite_cluster_id"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(
                    f"⚠️ Missing required columns in {regime_file.name}: {missing_columns}"
                )
                return False

            # Check for feature columns (should have more than just timestamp and cluster_id)
            feature_columns = [col for col in df.columns if col not in required_columns]
            if len(feature_columns) < 5:  # Minimum number of features
                self.logger.warning(
                    f"⚠️ Insufficient features in {regime_file.name}: {len(feature_columns)} features"
                )
                return False

            # Check regime distribution
            unique_regimes = df["composite_cluster_id"].nunique()
            if unique_regimes < 2:
                self.logger.warning(
                    f"⚠️ Insufficient regime diversity in {regime_file.name}: {unique_regimes} regimes"
                )
                return False

            self.logger.info(
                f"✅ Regime file validated: {len(df)} rows, {len(feature_columns)} features, {unique_regimes} regimes"
            )
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error validating regime file {regime_file}: {e}")
            return False

    @validate_file_operation
    def _validate_statistics_file(self, stats_file: Path) -> bool:
        """Validate a regime statistics file."""
        try:
            self.logger.info(f"📁 Validating statistics file: {stats_file.name}")

            # Load and validate the statistics file
            with open(stats_file, "r") as f:
                stats = json.load(f)

            # Check if stats is a dictionary
            if not isinstance(stats, dict):
                self.logger.warning(f"⚠️ Statistics file is not a valid JSON object: {stats_file.name}")
                return False

            # Check for required fields in each regime
            required_fields = ["total_rows", "date_range", "feature_count"]
            for regime_name, regime_stats in stats.items():
                missing_fields = [field for field in required_fields if field not in regime_stats]
                if missing_fields:
                    self.logger.warning(
                        f"⚠️ Missing required fields for regime {regime_name}: {missing_fields}"
                    )
                    return False

                # Validate total_rows
                total_rows = regime_stats.get("total_rows", 0)
                if total_rows < 50:  # Minimum data requirement
                    self.logger.warning(
                        f"⚠️ Insufficient data for regime {regime_name}: {total_rows} rows"
                    )
                    return False

            self.logger.info(f"✅ Statistics file validated: {len(stats)} regimes")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error validating statistics file {stats_file}: {e}")
            return False


@validate_step2_operation
def step4_regime_data_splitting_validator(
    symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any], config: dict[str, Any]
) -> bool:
    """Step 4: Regime Data Splitting Validator.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        training_input: Training input data
        config: Configuration dictionary

    Returns:
        bool: True if validation passes
    """
    logger.info("🔍 Starting Step 4: Regime Data Splitting validation")

    try:
        validator = Step4RegimeDataSplittingValidator(config)
        result = validator.validate_step4_regime_data_splitting(
            symbol, exchange, data_dir, training_input
        )

        if result:
            logger.info("✅ Step 4: Regime Data Splitting validation passed")
            return True
        else:
            logger.warning("⚠️ Step 4: Regime Data Splitting validation failed")
            return False

    except Exception as e:
        logger.exception(f"❌ Step 4 validation failed: {e}")
        return False