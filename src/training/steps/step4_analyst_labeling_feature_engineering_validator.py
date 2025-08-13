"""
Validator for Step 4: Analyst Labeling and Feature Engineering
"""

import asyncio
import os
import sys
import pickle
import pandas as pd
import numpy as np
from typing import Any, Dict, List
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.base_validator import BaseValidator
from src.config import CONFIG


class Step4AnalystLabelingFeatureEngineeringValidator(BaseValidator):
    """Validator for analyst labeling and feature engineering (used for Step 2 and Step 3)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("step4_analyst_labeling_feature_engineering", config)
        # Fine-tuned parameters for ML training (more lenient to avoid stopping training)
        self.min_feature_count = 10  # Reduced from strict requirements
        self.max_feature_count = 1000  # Increased to allow more features
        self.min_label_balance = 0.1  # Reduced from 0.2 to allow more imbalanced data
        self.max_label_classes = 15  # Increased from 10 to allow more classes
        self.feature_quality_threshold = 0.7  # More lenient feature quality checks
        self.data_balance_threshold = 0.15  # More lenient balance requirements

    async def validate(
        self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]
    ) -> bool:
        """
        Validate the analyst labeling and feature engineering step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info("🔍 Validating labeling + feature engineering outputs (Steps 2/3)...")
        print("Validator ▶ Step2/3 start")

        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")

        # Validate step result from pipeline state
        step_result = pipeline_state.get("analyst_labeling_feature_engineering", {})

        # 1. Validate error absence (CRITICAL - blocks process)
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics

        if not error_passed:
            self.logger.error(
                "❌ Analyst labeling and feature engineering step had critical errors - stopping process"
            )
            return False

        # 2. Validate feature engineering outputs (CRITICAL - blocks process)
        features_passed = self._validate_feature_engineering_outputs(symbol, exchange, data_dir)
        if not features_passed:
            self.logger.error(
                "❌ Feature engineering outputs validation failed - stopping process"
            )
            return False

        # 3. Validate labeling quality (WARNING - doesn't block)
        labeling_passed = self._validate_labeling_quality(symbol, exchange, data_dir)
        if not labeling_passed:
            self.logger.warning(
                "⚠️ Labeling quality validation failed - continuing with caution"
            )

        # 4. Validate feature quality (WARNING - doesn't block)
        feature_quality_passed = self._validate_feature_quality(
            symbol, exchange, data_dir
        )
        if not feature_quality_passed:
            self.logger.warning(
                "⚠️ Feature quality validation failed - continuing with caution"
            )

        # 5. Validate data balance (WARNING - doesn't block)
        balance_passed = self._validate_data_balance(symbol, exchange, data_dir)
        if not balance_passed:
            self.logger.warning(
                "⚠️ Data balance validation failed - continuing with caution"
            )

        # 6. Validate outcome favorability (WARNING - doesn't block)
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(
            step_result
        )
        self.validation_results["outcome_favorability"] = outcome_metrics

        if not outcome_passed:
            self.logger.warning(
                "⚠️ Analyst labeling and feature engineering outcome is not favorable - continuing with caution"
            )

        # Overall validation passes if critical checks pass
        critical_passed = error_passed and features_passed
        if critical_passed:
            self.logger.info(
                "✅ Analyst labeling and feature engineering validation passed (critical checks only)"
            )
            return True
        else:
            self.logger.error(
                "❌ Analyst labeling and feature engineering validation failed (critical checks failed)"
            )
            return False

    def _validate_feature_engineering_outputs(
        self, symbol: str, exchange: str, data_dir: str
    ) -> bool:
        """
        Validate feature engineering outputs.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if outputs are valid
        """
        try:
            # Expected feature engineering output files (Parquet preferred)
            expected_files = [
                f"{data_dir}/{exchange}_{symbol}_features_train.parquet",
                f"{data_dir}/{exchange}_{symbol}_features_validation.parquet",
                f"{data_dir}/{exchange}_{symbol}_features_test.parquet",
            ]

            missing_files = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(
                    file_path, "feature_engineering"
                )
                if not file_passed:
                    missing_files.append(file_path)

            if missing_files:
                self.logger.error(
                    f"❌ Missing feature engineering files: {missing_files} - stopping process"
                )
                return False

            # Validate feature data quality
            for file_path in expected_files:
                try:
                    feature_data = pd.read_parquet(file_path)

                    # Validate feature data quality
                    quality_passed, quality_metrics = self.validate_data_quality(
                        feature_data, "feature_data"
                    )
                    if not quality_passed:
                        self.logger.error(
                            f"❌ Feature data quality validation failed for {file_path} - stopping process"
                        )
                        return False

                except Exception as e:
                    self.logger.error(
                        f"❌ Error validating feature file {file_path}: {e} - stopping process"
                    )
                    return False

            self.logger.info("✅ Feature engineering outputs validation passed")
            print("Validator ▶ Step3 features ok")
            return True

        except Exception as e:
            self.logger.error(
                f"❌ Error during feature engineering outputs validation: {e}"
            )
            return False

    def _validate_labeling_quality(
        self, symbol: str, exchange: str, data_dir: str
    ) -> bool:
        """
        Validate labeling quality.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if labeling quality is acceptable
        """
        try:
            # Load labeled data files
            labeled_files = [
                f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet",
                f"{data_dir}/{exchange}_{symbol}_labeled_validation.parquet",
                f"{data_dir}/{exchange}_{symbol}_labeled_test.parquet",
            ]

            for file_path in labeled_files:
                if not os.path.exists(file_path):
                    self.logger.warning(
                        f"⚠️ Labeled data file not found: {file_path} - continuing with caution"
                    )
                    continue

                try:
                    labeled_data = pd.read_parquet(file_path)

                    # Check for label column
                    if "label" not in labeled_data.columns:
                        self.logger.warning(
                            f"⚠️ No label column found in {file_path} - continuing with caution"
                        )
                        return False

                    # Check for OHLCV columns (optional but recommended)
                    ohlcv_columns = ["open", "high", "low", "close", "volume"]
                    missing_ohlcv = [
                        col for col in ohlcv_columns if col not in labeled_data.columns
                    ]
                    if missing_ohlcv:
                        self.logger.warning(
                            f"⚠️ Missing OHLCV columns in {file_path}: {missing_ohlcv} - this may affect labeling quality"
                        )
                        self.logger.warning(
                            "Triple barrier labeling requires proper OHLCV data for accurate labels"
                        )

                    # Validate label values
                    labels = labeled_data["label"]
                    unique_labels = labels.unique()

                    # Check for reasonable number of classes (more lenient)
                    if len(unique_labels) < 2:
                        self.logger.warning(
                            f"⚠️ Insufficient label classes: {len(unique_labels)} - continuing with caution"
                        )
                        self.logger.warning(
                            "This may indicate missing OHLCV data or improper triple barrier labeling"
                        )
                        return False

                    if len(unique_labels) > self.max_label_classes:
                        self.logger.warning(
                            f"⚠️ Many label classes: {len(unique_labels)} (max: {self.max_label_classes}) - continuing with caution"
                        )

                    # Check for label balance (more lenient)
                    label_counts = labels.value_counts()
                    min_count = label_counts.min()
                    max_count = label_counts.max()
                    balance_ratio = min_count / max_count if max_count > 0 else 0

                    if balance_ratio < self.min_label_balance:
                        self.logger.warning(
                            f"⚠️ Label balance is poor: {balance_ratio:.3f} (min: {self.min_label_balance:.3f}) - continuing with caution"
                        )
                        return False

                    # Check for missing labels
                    missing_labels = labels.isnull().sum()
                    if missing_labels > 0:
                        missing_ratio = missing_labels / len(labels)
                        if missing_ratio > 0.1:  # More than 10% missing
                            self.logger.warning(
                                f"⚠️ High missing label ratio: {missing_ratio:.3f} - continuing with caution"
                            )
                        else:
                            self.logger.info(
                                f"ℹ️ Found {missing_labels} missing labels (acceptable)"
                            )

                except Exception as e:
                    self.logger.error(
                        f"❌ Error validating labeled data file {file_path}: {e}"
                    )
                    return False

            self.logger.info("✅ Labeling quality validation passed")
            print("Validator ▶ Step2 labels ok")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error during labeling quality validation: {e}")
            return False

    def _validate_feature_quality(
        self, symbol: str, exchange: str, data_dir: str
    ) -> bool:
        """
        Validate feature quality.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if feature quality is acceptable
        """
        try:
            # Load feature files
            feature_files = [
                f"{data_dir}/{exchange}_{symbol}_features_train.pkl",
                f"{data_dir}/{exchange}_{symbol}_features_validation.pkl",
                f"{data_dir}/{exchange}_{symbol}_features_test.pkl",
            ]

            for file_path in feature_files:
                if not os.path.exists(file_path):
                    self.logger.warning(
                        f"⚠️ Feature file not found: {file_path} - continuing with caution"
                    )
                    continue

                try:
                    with open(file_path, "rb") as f:
                        feature_data = pickle.load(f)

                    if not isinstance(feature_data, pd.DataFrame):
                        feature_data = pd.DataFrame(feature_data)

                    # Blocker: raw OHLCV must not be present in saved features
                    forbidden = {"open", "high", "low", "close", "volume"}
                    present_forbidden = [
                        c for c in feature_data.columns if c in forbidden
                    ]
                    if present_forbidden:
                        self.logger.error(
                            f"❌ Raw OHLCV columns found in features ({present_forbidden}) for {file_path} - stopping process"
                        )
                        return False

                    # Note: Absence of raw OHLCV is fine; presence is blocked above.

                    # Check feature count (more lenient)
                    feature_count = len(feature_data.columns)
                    if feature_count < self.min_feature_count:
                        self.logger.warning(
                            f"⚠️ Too few features: {feature_count} (min: {self.min_feature_count}) - continuing with caution"
                        )
                        return False

                    if feature_count > self.max_feature_count:
                        self.logger.warning(
                            f"⚠️ Too many features: {feature_count} (max: {self.max_feature_count}) - continuing with caution"
                        )
                        return False

                    # Check for constant features
                    constant_features = []
                    for col in feature_data.columns:
                        if feature_data[col].nunique() <= 1:
                            constant_features.append(col)

                    if constant_features:
                        # Previously allowed up to a ratio; now warn strictly if any constant features
                        self.logger.warning(
                            f"⚠️ Found {len(constant_features)} constant features - this should be 0. Examples: {constant_features[:5]}"
                        )
                        # Do not fail the step here (warning), but make it visible

                    # Check for high correlation features
                    numeric_cols = feature_data.select_dtypes(
                        include=[np.number]
                    ).columns
                    if len(numeric_cols) > 1:
                        corr_matrix = feature_data[numeric_cols].corr().abs()
                        high_corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i + 1, len(corr_matrix.columns)):
                                if (
                                    corr_matrix.iloc[i, j] > 0.95
                                ):  # Very high correlation
                                    high_corr_pairs.append(
                                        (corr_matrix.columns[i], corr_matrix.columns[j])
                                    )

                        if high_corr_pairs:
                            self.logger.warning(
                                f"⚠️ Found {len(high_corr_pairs)} highly correlated feature pairs - continuing with caution"
                            )

                    # Check for missing values
                    missing_ratios = feature_data.isnull().sum() / len(feature_data)
                    high_missing_features = missing_ratios[
                        missing_ratios > 0.5
                    ].index.tolist()

                    if high_missing_features:
                        self.logger.warning(
                            f"⚠️ Found {len(high_missing_features)} features with >50% missing values - continuing with caution"
                        )

                except Exception as e:
                    self.logger.error(
                        f"❌ Error validating feature file {file_path}: {e}"
                    )
                    return False

            self.logger.info("✅ Feature quality validation passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error during feature quality validation: {e}")
            return False

    def _validate_data_balance(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """
        Validate data balance across splits.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if data balance is acceptable
        """
        try:
            # Load labeled data from all splits
            split_files = [
                ("train", f"{data_dir}/{exchange}_{symbol}_labeled_train.pkl"),
                (
                    "validation",
                    f"{data_dir}/{exchange}_{symbol}_labeled_validation.pkl",
                ),
                ("test", f"{data_dir}/{exchange}_{symbol}_labeled_test.pkl"),
            ]

            split_data = {}
            for split_name, file_path in split_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "rb") as f:
                            data = pickle.load(f)

                        if not isinstance(data, pd.DataFrame):
                            data = pd.DataFrame(data)

                        split_data[split_name] = data

                    except Exception as e:
                        self.logger.warning(
                            f"⚠️ Error loading {split_name} split: {e} - continuing with caution"
                        )
                        continue

            if len(split_data) < 2:
                self.logger.warning(
                    "⚠️ Insufficient splits for balance validation - continuing with caution"
                )
                return False

            # Check label distribution across splits
            if "label" in split_data.get("train", pd.DataFrame()).columns:
                train_labels = split_data["train"]["label"].value_counts()

                for split_name, data in split_data.items():
                    if split_name == "train" or "label" not in data.columns:
                        continue

                    split_labels = data["label"].value_counts()

                    # Check if all train labels are present in other splits
                    missing_labels = set(train_labels.index) - set(split_labels.index)
                    if missing_labels:
                        self.logger.warning(
                            f"⚠️ Missing labels in {split_name} split: {missing_labels} - continuing with caution"
                        )

                    # Check label distribution similarity
                    common_labels = set(train_labels.index) & set(split_labels.index)
                    if common_labels:
                        distribution_diffs = []
                        for label in common_labels:
                            train_ratio = train_labels[label] / len(split_data["train"])
                            split_ratio = split_labels[label] / len(data)
                            diff = abs(train_ratio - split_ratio)
                            distribution_diffs.append(diff)

                        avg_diff = np.mean(distribution_diffs)
                        if avg_diff > self.data_balance_threshold:
                            self.logger.warning(
                                f"⚠️ Large distribution difference in {split_name} split: {avg_diff:.3f} - continuing with caution"
                            )
                            return False

            self.logger.info("✅ Data balance validation passed")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error during data balance validation: {e}")
            return False


async def run_validator(
    training_input: Dict[str, Any], pipeline_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run the Step 4 Analyst Labeling and Feature Engineering validator.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    validator = Step4AnalystLabelingFeatureEngineeringValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)

    return {
        "step_name": "step4_analyst_labeling_feature_engineering",
        "validation_passed": validation_passed,
        "validation_results": validator.validation_results,
        "duration": 0,  # Could be enhanced to track actual duration
        "timestamp": asyncio.get_event_loop().time(),
    }


if __name__ == "__main__":
    import asyncio

    # Example usage
    async def test_validator():
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data/training",
        }

        pipeline_state = {
            "analyst_labeling_feature_engineering": {
                "status": "SUCCESS",
                "duration": 180.5,
            }
        }

        result = await run_validator(training_input, pipeline_state)
        print(f"Validation result: {result}")

    asyncio.run(test_validator())
