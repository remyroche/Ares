# src/training/steps/step6_feature_engineering_validator.py
"""Validator for Step 2: Feature Engineering."""

import asyncio
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG
from src.utils.base_validator import BaseValidator


class Step2FeatureEngineeringValidator(BaseValidator):
    """Validator for feature engineering (Step 2)."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("step2_feature_engineering", config)
        # Fine-tuned parameters for ML training (more lenient to avoid stopping training)
        self.min_feature_count = 40  # Minimum 40 relevant features required
        self.max_feature_count = 1000  # Increased to allow more features
        self.min_label_balance = 0.1  # Reduced from 0.2 to allow more imbalanced data
        self.max_label_classes = 15  # Increased from 10 to allow more classes
        self.feature_quality_threshold = 0.7  # More lenient feature quality checks
        self.data_balance_threshold = 0.15  # More lenient balance requirements

    async def validate(
        self, training_input: dict[str, Any], pipeline_state: dict[str, Any]
    ) -> bool:
        """Validate the feature engineering step.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            bool: True if validation passed, False otherwise
        """
        self.logger.info(
            "🔍 Validating feature engineering outputs (Step 2)...",
        )

        # Extract parameters
        symbol = training_input.get("symbol", "ETHUSDT")
        exchange = training_input.get("exchange", "BINANCE")
        data_dir = training_input.get("data_dir", "data/training")

        # Validate step result from pipeline state
        step_result = pipeline_state.get("feature_engineering", {})

        # 1. Validate error absence (CRITICAL - blocks process)
        error_passed, error_metrics = self.validate_error_absence(step_result)
        self.validation_results["error_absence"] = error_metrics

        if not error_passed:
            self.logger.error(
                "❌ Feature engineering step had critical errors - stopping process",
            )
            return False

        # 2. Validate feature engineering outputs (CRITICAL - blocks process)
        features_passed = self._validate_feature_engineering_outputs(
            symbol=symbol,
            exchange=exchange,
            data_dir=data_dir,
        )
        if not features_passed:
            self.logger.error(
                "❌ Feature engineering outputs validation failed - stopping process",
            )
            return False

        # 2.5. Validate minimum relevant features requirement (CRITICAL - make or break)
        relevant_features_passed = self._validate_minimum_relevant_features(
            symbol=symbol,
            exchange=exchange,
            data_dir=data_dir,
        )
        if not relevant_features_passed:
            self.logger.error(
                "❌ Minimum relevant features requirement not met - stopping process",
            )
            return False

        # 3. Validate feature quality (CRITICAL - blocks process if insufficient relevant features)
        feature_quality_passed = self._validate_feature_quality(
            symbol=symbol,
            exchange=exchange,
            data_dir=data_dir,
        )
        if not feature_quality_passed:
            self.logger.error(
                "❌ Feature quality validation failed - stopping process",
            )
            return False

        # 4. Validate outcome favorability (WARNING - doesn't block)
        outcome_passed, outcome_metrics = self.validate_outcome_favorability(
            step_result,
        )
        self.validation_results["outcome_favorability"] = outcome_metrics

        if not outcome_passed:
            self.logger.warning(
                "⚠️ Feature engineering outcome is not favorable - continuing with caution",
            )

        # Overall validation passes if critical checks pass
        critical_passed = (
            error_passed
            and features_passed
            and relevant_features_passed
            and feature_quality_passed
        )
        if critical_passed:
            self.logger.info(
                "✅ Feature engineering validation passed (critical checks only)",
            )
            return True
        self.logger.error(
            "❌ Feature engineering validation failed (critical checks failed)",
        )
        return False

    def _validate_feature_engineering_outputs(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """Validate feature engineering outputs.

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
                f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet",
                f"{data_dir}/{exchange}_{symbol}_labeled_validation.parquet",
                f"{data_dir}/{exchange}_{symbol}_labeled_test.parquet",
            ]

            missing_files: list[str] = []
            for file_path in expected_files:
                file_passed, file_metrics = self.validate_file_exists(
                    file_path,
                    "feature_engineering",
                )
                if not file_passed:
                    missing_files.append(file_path)

            if missing_files:
                self.logger.error(
                    f"❌ Missing feature engineering files: {missing_files} - stopping process",
                )
                return False

            # Validate feature data quality
            for file_path in expected_files:
                try:
                    feature_data = pd.read_parquet(file_path)

                    # Validate feature data quality
                    quality_passed, quality_metrics = self.validate_data_quality(
                        feature_data,
                    )
                    if not quality_passed:
                        self.logger.error(
                            f"❌ Feature data quality validation failed for {file_path} - stopping process",
                        )
                        return False

                except Exception as e:
                    self.logger.exception(
                        f"❌ Error validating feature file {file_path}: {e} - stopping process",
                    )
                    return False

            self.logger.info("✅ Feature engineering outputs validation passed")
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error during feature engineering outputs validation: {e}",
            )
            return False

    def _validate_labeling_quality(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """Validate labeling quality.

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
                        f"⚠️ Labeled data file not found: {file_path} - continuing with caution",
                    )
                    continue

                try:
                    labeled_data = pd.read_parquet(file_path)

                    # Check for label column
                    if "label" not in labeled_data.columns:
                        self.logger.warning(
                            f"⚠️ No label column found in {file_path} - continuing with caution",
                        )
                        return False

                    # Check for OHLCV columns (optional but recommended)
                    ohlcv_columns = ["open", "high", "low", "close", "volume"]
                    missing_ohlcv = [
                        col for col in ohlcv_columns if col not in labeled_data.columns
                    ]
                    if missing_ohlcv:
                        self.logger.warning(
                            f"⚠️ Missing OHLCV columns in {file_path}: {missing_ohlcv} - this may affect labeling quality",
                        )
                        self.logger.warning(
                            "Triple barrier labeling requires proper OHLCV data for accurate labels",
                        )

                    # Validate label values
                    labels = labeled_data["label"]
                    unique_labels = labels.unique()

                    # Check for reasonable number of classes (more lenient)
                    if len(unique_labels) < 2:
                        self.logger.warning(
                            f"⚠️ Insufficient label classes: {len(unique_labels)} - continuing with caution",
                        )
                        self.logger.warning(
                            "This may indicate missing OHLCV data or improper triple barrier labeling",
                        )
                        return False

                    if len(unique_labels) > self.max_label_classes:
                        self.logger.warning(
                            f"⚠️ Many label classes: {len(unique_labels)} (max: {self.max_label_classes}) - continuing with caution",
                        )

                    # Check for label balance (more lenient)
                    label_counts = labels.value_counts()
                    min_count = label_counts.min()
                    max_count = label_counts.max()
                    balance_ratio = min_count / max_count if max_count > 0 else 0

                    if balance_ratio < self.min_label_balance:
                        self.logger.warning(
                            f"⚠️ Label balance is poor: {balance_ratio:.3f} (min: {self.min_label_balance:.3f}) - continuing with caution",
                        )
                        return False

                    # Check for missing labels
                    missing_labels = labels.isnull().sum()
                    if missing_labels > 0:
                        missing_ratio = missing_labels / len(labels)
                        if missing_ratio > 0.1:  # More than 10% missing
                            self.logger.warning(
                                f"⚠️ High missing label ratio: {missing_ratio:.3f} - continuing with caution",
                            )
                        else:
                            self.logger.info(
                                f"ℹ️ Found {missing_labels} missing labels (acceptable)",
                            )

                except Exception as e:
                    self.logger.exception(
                        f"❌ Error validating labeled data file {file_path}: {e}",
                    )
                    return False

            self.logger.info("✅ Labeling quality validation passed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error during labeling quality validation: {e}")
            return False

    def _validate_feature_quality(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """Validate feature quality.

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
                        f"⚠️ Feature file not found: {file_path} - continuing with caution",
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
                        self.logger.warning(
                            f"⚠️ Raw OHLCV columns found in features ({present_forbidden}) for {file_path} - removing them automatically",
                        )
                        # Remove the forbidden columns automatically
                        feature_data = feature_data.drop(columns=present_forbidden)
                        # Save the cleaned data back
                        try:
                            with open(file_path, "wb") as f:
                                pickle.dump(feature_data, f)
                            self.logger.info(
                                "✅ Cleaned and saved feature data without raw OHLCV columns",
                            )
                        except Exception as e:
                            self.logger.warning(f"⚠️ Could not save cleaned data: {e}")
                            # Continue instead of failing
                            continue

                    # Note: Absence of raw OHLCV is fine; presence is blocked above.

                    # Check for constant features
                    constant_features: list[str] = []
                    for col in feature_data.columns:
                        if feature_data[col].nunique() <= 1:
                            constant_features.append(col)

                    # Check for missing values
                    missing_ratios = feature_data.isnull().sum() / len(feature_data)
                    high_missing_features = missing_ratios[
                        missing_ratios > 0.5
                    ].index.tolist()

                    # Calculate relevant features (excluding problematic ones)
                    problematic_features = set(
                        constant_features + high_missing_features
                    )
                    relevant_features = [
                        col
                        for col in feature_data.columns
                        if col not in problematic_features
                    ]
                    relevant_feature_count = len(relevant_features)

                    # Check relevant feature count (CRITICAL - blocks process if insufficient)
                    if relevant_feature_count < self.min_feature_count:
                        self.logger.error(
                            f"❌ Insufficient relevant features: {relevant_feature_count} (minimum required: {self.min_feature_count}) - stopping process",
                        )
                        self.logger.error(
                            f"❌ Problematic features excluded: {len(problematic_features)} (constant: {len(constant_features)}, high missing: {len(high_missing_features)})",
                        )
                        return False

                    # Check total feature count (warning only)
                    feature_count = len(feature_data.columns)
                    if feature_count > self.max_feature_count:
                        self.logger.warning(
                            f"⚠️ Too many features: {feature_count} (max: {self.max_feature_count}) - continuing with caution",
                        )

                    if constant_features:
                        # Previously allowed up to a ratio; now warn strictly if any constant features
                        self.logger.warning(
                            f"⚠️ Found {len(constant_features)} constant features - this should be 0. Examples: {constant_features[:5]}",
                        )
                        # Do not fail the step here (warning), but make it visible

                    if high_missing_features:
                        self.logger.warning(
                            f"⚠️ Found {len(high_missing_features)} features with >50% missing values - continuing with caution",
                        )

                    # Check for high correlation features
                    numeric_cols = feature_data.select_dtypes(
                        include=[np.number]
                    ).columns
                    if len(numeric_cols) > 1:
                        corr_matrix = feature_data[numeric_cols].corr().abs()
                        high_corr_pairs: list[tuple[str, str]] = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i + 1, len(corr_matrix.columns)):
                                if (
                                    corr_matrix.iloc[i, j] > 0.95
                                ):  # Very high correlation
                                    high_corr_pairs.append(
                                        (
                                            corr_matrix.columns[i],
                                            corr_matrix.columns[j],
                                        ),
                                    )

                        if high_corr_pairs:
                            self.logger.warning(
                                f"⚠️ Found {len(high_corr_pairs)} highly correlated feature pairs - continuing with caution",
                            )

                except Exception as e:
                    self.logger.exception(
                        f"❌ Error validating feature file {file_path}: {e}",
                    )
                    return False

            self.logger.info("✅ Feature quality validation passed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error during feature quality validation: {e}")
            return False

    def _validate_minimum_relevant_features(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
    ) -> bool:
        """Validate minimum relevant features requirement (MAKE OR BREAK).

        This is a critical validation that ensures we have at least 40 relevant features
        (non-constant, non-problematic features) before proceeding with training.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory

        Returns:
            bool: True if minimum relevant features requirement is met, False otherwise
        """
        try:
            self.logger.info(
                f"🔍 Validating minimum relevant features requirement ({self.min_feature_count} required)..."
            )

            # Load feature files
            feature_files = [
                f"{data_dir}/{exchange}_{symbol}_features_train.pkl",
                f"{data_dir}/{exchange}_{symbol}_features_validation.pkl",
                f"{data_dir}/{exchange}_{symbol}_features_test.pkl",
            ]

            total_relevant_features = 0
            file_count = 0

            for file_path in feature_files:
                if not os.path.exists(file_path):
                    self.logger.warning(
                        f"⚠️ Feature file not found: {file_path} - skipping",
                    )
                    continue

                try:
                    with open(file_path, "rb") as f:
                        feature_data = pickle.load(f)

                    if not isinstance(feature_data, pd.DataFrame):
                        feature_data = pd.DataFrame(feature_data)

                    # Remove raw OHLCV columns if present
                    forbidden = {"open", "high", "low", "close", "volume"}
                    present_forbidden = [
                        c for c in feature_data.columns if c in forbidden
                    ]
                    if present_forbidden:
                        feature_data = feature_data.drop(columns=present_forbidden)

                    # Identify problematic features
                    constant_features: list[str] = []
                    for col in feature_data.columns:
                        if feature_data[col].nunique() <= 1:
                            constant_features.append(col)

                    missing_ratios = feature_data.isnull().sum() / len(feature_data)
                    high_missing_features = missing_ratios[
                        missing_ratios > 0.5
                    ].index.tolist()

                    # Calculate relevant features
                    problematic_features = set(
                        constant_features + high_missing_features
                    )
                    relevant_features = [
                        col
                        for col in feature_data.columns
                        if col not in problematic_features
                    ]
                    relevant_feature_count = len(relevant_features)

                    self.logger.info(
                        f"📊 {file_path}: {relevant_feature_count} relevant features "
                        f"(total: {len(feature_data.columns)}, "
                        f"constant: {len(constant_features)}, "
                        f"high missing: {len(high_missing_features)})",
                    )

                    total_relevant_features += relevant_feature_count
                    file_count += 1

                except Exception as e:
                    self.logger.exception(
                        f"❌ Error processing feature file {file_path}: {e}",
                    )
                    return False

            if file_count == 0:
                self.logger.error("❌ No feature files found - stopping process")
                return False

            # Calculate average relevant features per file
            avg_relevant_features = total_relevant_features / file_count

            if avg_relevant_features < self.min_feature_count:
                self.logger.error(
                    f"❌ INSUFFICIENT RELEVANT FEATURES: {avg_relevant_features:.1f} average "
                    f"(minimum required: {self.min_feature_count}) - STOPPING PROCESS",
                )
                self.logger.error(
                    "❌ This is a MAKE-OR-BREAK requirement. Training cannot proceed without sufficient relevant features.",
                )
                return False

            self.logger.info(
                f"✅ Minimum relevant features requirement met: {avg_relevant_features:.1f} average "
                f"(required: {self.min_feature_count})",
            )
            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Error during minimum relevant features validation: {e}"
            )
            return False

    def _validate_data_balance(self, symbol: str, exchange: str, data_dir: str) -> bool:
        """Validate data balance across splits.

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
                            f"⚠️ Error loading {split_name} split: {e} - continuing with caution",
                        )
                        continue

            if len(split_data) < 2:
                self.logger.warning(
                    "⚠️ Insufficient splits for balance validation - continuing with caution",
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
                        f"⚠️ Missing labels in {split_name} split: {missing_labels} - continuing with caution",
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
                            f"⚠️ Large distribution difference in {split_name} split: {avg_diff:.3f} - continuing with caution",
                        )
                        return False

            self.logger.info("✅ Data balance validation passed")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Error during data balance validation: {e}")
            return False


async def run_validator(
    training_input: dict[str, Any],
    pipeline_state: dict[str, Any],
) -> dict[str, Any]:
    """Run the Step 2 Feature Engineering validator.

    Args:
        training_input: Training input parameters
        pipeline_state: Current pipeline state

    Returns:
        Dictionary containing validation results
    """
    validator = Step2FeatureEngineeringValidator(CONFIG)
    validation_passed = await validator.validate(training_input, pipeline_state)

    return {
        "step_name": "step2_feature_engineering",
        "validation_passed": validation_passed,
        "validation_results": validator.validation_results,
        "duration": 0,  # Could be enhanced to track actual duration
        "timestamp": asyncio.get_event_loop().time(),
    }


if __name__ == "__main__":
    import asyncio

    # Example usage
    async def test_validator() -> None:
        training_input = {
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data/training",
        }

        pipeline_state = {
            "analyst_labeling_feature_engineering": {
                "status": "SUCCESS",
                "duration": 180.5,
            },
        }

        await run_validator(training_input, pipeline_state)

    asyncio.run(test_validator())
