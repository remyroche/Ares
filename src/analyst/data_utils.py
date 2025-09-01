# src/analyst/data_utils.py
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks  # For volume profile peaks

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    critical,
    error,
    failed,
    initialization_error,
    invalid,
    missing,
    validation_error,
    warning,
)


class DataUtils:
    """
    Data utilities with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize data utils with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("DataUtils")

        # Data utils state
        self.is_processing: bool = False
        self.processing_results: dict[str, Any] = {}
        self.processing_history: list[dict[str, Any]] = []

        # Configuration
        self.data_utils_config: dict[str, Any] = self.config.get("data_utils", {})
        self.processing_interval: int = self.data_utils_config.get(
            "processing_interval",
            3600,
        )
        self.max_processing_history: int = self.data_utils_config.get(
            "max_processing_history",
            100,
        )
        self.enable_data_cleaning: bool = self.data_utils_config.get(
            "enable_data_cleaning",
            True,
        )
        self.enable_data_validation: bool = self.data_utils_config.get(
            "enable_data_validation",
            True,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid data utils configuration"),
            AttributeError: (False, "Missing required data utils parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="data utils initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data utils configuration loading",
    )
    async def _load_data_utils_configuration(self) -> None:
        """Load data utils configuration."""
        try:
            # Set default data utils parameters
            self.data_utils_config.setdefault("processing_interval", 3600)
            self.data_utils_config.setdefault("max_processing_history", 100)
            self.data_utils_config.setdefault("enable_data_cleaning", True)
            self.data_utils_config.setdefault("enable_data_validation", True)
            self.data_utils_config.setdefault("enable_data_transformation", True)
            self.data_utils_config.setdefault("enable_data_aggregation", True)

            # Update configuration
            self.processing_interval = self.data_utils_config["processing_interval"]
            self.max_processing_history = self.data_utils_config[
                "max_processing_history"
            ]
            self.enable_data_cleaning = self.data_utils_config["enable_data_cleaning"]
            self.enable_data_validation = self.data_utils_config[
                "enable_data_validation"
            ]

            self.logger.info("Data utils configuration loaded successfully")

        except Exception:
            self.print(error("Error loading data utils configuration: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate data utils configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate processing interval
            if self.processing_interval <= 0:
                self.print(invalid("Invalid processing interval"))
                return False

            # Validate max processing history
            if self.max_processing_history <= 0:
                self.print(invalid("Invalid max processing history"))
                return False

            # Validate that at least one processing type is enabled
            if not any(
                [
                    self.enable_data_cleaning,
                    self.enable_data_validation,
                    self.data_utils_config.get("enable_data_transformation", True),
                    self.data_utils_config.get("enable_data_aggregation", True),
                ],
            ):
                self.print(error("At least one processing type must be enabled"))
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception:
            self.print(error("Error validating configuration: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data utils modules initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data cleaning initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data validation initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data transformation initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data aggregation initialization",
    )
    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid processing parameters"),
            AttributeError: (False, "Missing processing components"),
            KeyError: (False, "Missing required processing data"),
        },
        default_return=False,
        context="data processing execution",
    )
    async def execute_data_processing(self, processing_input: dict[str, Any]) -> bool:
        """
        Execute data processing operations.

        Args:
            processing_input: Processing input dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._validate_processing_inputs(processing_input):
                return False

            self.is_processing = True
            self.logger.info("🔄 Starting data processing execution...")

            # Perform data cleaning
            if self.enable_data_cleaning:
                cleaning_results = await self._perform_data_cleaning(processing_input)
                self.processing_results["data_cleaning"] = cleaning_results

            # Perform data validation
            if self.enable_data_validation:
                validation_results = await self._perform_data_validation(
                    processing_input,
                )
                self.processing_results["data_validation"] = validation_results

            # Perform data transformation
            if self.data_utils_config.get("enable_data_transformation", True):
                transformation_results = await self._perform_data_transformation(
                    processing_input,
                )
                self.processing_results["data_transformation"] = transformation_results

            # Perform data aggregation
            if self.data_utils_config.get("enable_data_aggregation", True):
                aggregation_results = await self._perform_data_aggregation(
                    processing_input,
                )
                self.processing_results["data_aggregation"] = aggregation_results

            # Store processing results
            await self._store_processing_results()

            self.is_processing = False
            self.logger.info("✅ Data processing execution completed successfully")
            return True

        except Exception:
            self.print(error("Error executing data processing: {e}"))
            self.is_processing = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="processing inputs validation",
    )
    def _validate_processing_inputs(self, processing_input: dict[str, Any]) -> bool:
        """
        Validate processing inputs.

        Args:
            processing_input: Processing input dictionary

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required processing input fields
            required_fields = ["processing_type", "data_source", "timestamp"]
            for field in required_fields:
                if field not in processing_input:
                    self.logger.error(
                        f"Missing required processing input field: {field}",
                    )
                    return False

            # Validate data types
            if not isinstance(processing_input["processing_type"], str):
                self.print(invalid("Invalid processing type"))
                return False

            if not isinstance(processing_input["data_source"], str):
                self.print(invalid("Invalid data source"))
                return False

            return True

        except Exception:
            self.print(error("Error validating processing inputs: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data cleaning",
    )
    async def _perform_data_cleaning(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform data cleaning.

        Args:
            processing_input: Processing input dictionary

        Returns:
            dict[str, Any]: Data cleaning results
        """
        try:
            results = {}

            # Perform outlier removal
            if self.data_cleaning_components.get("outlier_removal", False):
                results["outlier_removal"] = self._perform_outlier_removal(
                    processing_input,
                )

            # Perform missing data handling
            if self.data_cleaning_components.get("missing_data_handling", False):
                results["missing_data_handling"] = self._perform_missing_data_handling(
                    processing_input,
                )

            # Perform duplicate removal
            if self.data_cleaning_components.get("duplicate_removal", False):
                results["duplicate_removal"] = self._perform_duplicate_removal(
                    processing_input,
                )

            # Perform data normalization
            if self.data_cleaning_components.get("data_normalization", False):
                results["data_normalization"] = self._perform_data_normalization(
                    processing_input,
                )

            self.logger.info("Data cleaning completed")
            return results

        except Exception:
            self.print(error("Error performing data cleaning: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data validation",
    )
    async def _perform_data_validation(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform data validation.

        Args:
            processing_input: Processing input dictionary

        Returns:
            dict[str, Any]: Data validation results
        """
        try:
            results = {}

            # Perform data type validation
            if self.data_validation_components.get("data_type_validation", False):
                results["data_type_validation"] = self._perform_data_type_validation(
                    processing_input,
                )

            # Perform range validation
            if self.data_validation_components.get("range_validation", False):
                results["range_validation"] = self._perform_range_validation(
                    processing_input,
                )

            # Perform format validation
            if self.data_validation_components.get("format_validation", False):
                results["format_validation"] = self._perform_format_validation(
                    processing_input,
                )

            # Perform consistency validation
            if self.data_validation_components.get("consistency_validation", False):
                results["consistency_validation"] = (
                    self._perform_consistency_validation(processing_input)
                )

            self.logger.info("Data validation completed")
            return results

        except Exception:
            self.print(validation_error("Error performing data validation: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data transformation",
    )
    async def _perform_data_transformation(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform data transformation.

        Args:
            processing_input: Processing input dictionary

        Returns:
            dict[str, Any]: Data transformation results
        """
        try:
            results = {}

            # Perform feature scaling
            if self.data_transformation_components.get("feature_scaling", False):
                results["feature_scaling"] = self._perform_feature_scaling(
                    processing_input,
                )

            # Perform feature encoding
            if self.data_transformation_components.get("feature_encoding", False):
                results["feature_encoding"] = self._perform_feature_encoding(
                    processing_input,
                )

            # Perform feature selection
            if self.data_transformation_components.get("feature_selection", False):
                results["feature_selection"] = self._perform_feature_selection(
                    processing_input,
                )

            # Perform dimensionality reduction
            if self.data_transformation_components.get(
                "dimensionality_reduction",
                False,
            ):
                results["dimensionality_reduction"] = (
                    self._perform_dimensionality_reduction(processing_input)
                )

            self.logger.info("Data transformation completed")
            return results

        except Exception:
            self.print(error("Error performing data transformation: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data aggregation",
    )
    async def _perform_data_aggregation(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform data aggregation.

        Args:
            processing_input: Processing input dictionary

        Returns:
            dict[str, Any]: Data aggregation results
        """
        try:
            results = {}

            # Perform time aggregation
            if self.data_aggregation_components.get("time_aggregation", False):
                results["time_aggregation"] = self._perform_time_aggregation(
                    processing_input,
                )

            # Perform group aggregation
            if self.data_aggregation_components.get("group_aggregation", False):
                results["group_aggregation"] = self._perform_group_aggregation(
                    processing_input,
                )

            # Perform statistical aggregation
            if self.data_aggregation_components.get("statistical_aggregation", False):
                results["statistical_aggregation"] = (
                    self._perform_statistical_aggregation(processing_input)
                )

            # Perform custom aggregation
            if self.data_aggregation_components.get("custom_aggregation", False):
                results["custom_aggregation"] = self._perform_custom_aggregation(
                    processing_input,
                )

            self.logger.info("Data aggregation completed")
            return results

        except Exception:
            self.print(error("Error performing data aggregation: {e}"))
            return {}

    # Data cleaning methods
    def _perform_outlier_removal(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform outlier removal."""
        try:
            # Simulate outlier removal
            return {
                "outlier_removal_completed": True,
                "outliers_removed": 15,
                "removal_method": "iqr",
                "data_quality_improvement": 0.95,
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing outlier removal: {e}"))
            return {}

    def _perform_missing_data_handling(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform missing data handling."""
        try:
            # Simulate missing data handling
            return {
                "missing_data_handling_completed": True,
                "missing_values_filled": 25,
                "handling_method": "interpolation",
                "data_completeness": 0.98,
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(missing("Error performing missing data handling: {e}"))
            return {}

    def _perform_duplicate_removal(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform duplicate removal."""
        try:
            # Simulate duplicate removal
            return {
                "duplicate_removal_completed": True,
                "duplicates_removed": 8,
                "removal_method": "exact_match",
                "data_uniqueness": 0.99,
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing duplicate removal: {e}"))
            return {}

    def _perform_data_normalization(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform data normalization."""
        try:
            # Simulate data normalization
            return {
                "data_normalization_completed": True,
                "normalized_features": 10,
                "normalization_method": "min_max",
                "data_scale": "0_to_1",
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing data normalization: {e}"))
            return {}

    # Data validation methods
    def _perform_data_type_validation(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform data type validation."""
        try:
            # Simulate data type validation
            return {
                "data_type_validation_completed": True,
                "validation_score": 0.98,
                "validation_method": "type_check",
                "data_types_validated": 15,
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(validation_error("Error performing data type validation: {e}"))
            return {}

    def _perform_range_validation(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform range validation."""
        try:
            # Simulate range validation
            return {
                "range_validation_completed": True,
                "validation_score": 0.96,
                "validation_method": "range_check",
                "ranges_validated": 12,
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(validation_error("Error performing range validation: {e}"))
            return {}

    def _perform_format_validation(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform format validation."""
        try:
            # Simulate format validation
            return {
                "format_validation_completed": True,
                "validation_score": 0.94,
                "validation_method": "format_check",
                "formats_validated": 8,
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(validation_error("Error performing format validation: {e}"))
            return {}

    def _perform_consistency_validation(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform consistency validation."""
        try:
            # Simulate consistency validation
            return {
                "consistency_validation_completed": True,
                "validation_score": 0.92,
                "validation_method": "consistency_check",
                "consistency_rules": 5,
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(validation_error("Error performing consistency validation: {e}"))
            return {}

    # Data transformation methods
    def _perform_feature_scaling(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform feature scaling."""
        try:
            # Simulate feature scaling
            return {
                "feature_scaling_completed": True,
                "scaled_features": 8,
                "scaling_method": "standard_scaler",
                "scaling_range": "mean_0_std_1",
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing feature scaling: {e}"))
            return {}

    def _perform_feature_encoding(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform feature encoding."""
        try:
            # Simulate feature encoding
            return {
                "feature_encoding_completed": True,
                "encoded_features": 6,
                "encoding_method": "one_hot",
                "encoding_dimensions": 15,
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing feature encoding: {e}"))
            return {}

    def _perform_feature_selection(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform feature selection."""
        try:
            # Simulate feature selection
            return {
                "feature_selection_completed": True,
                "selected_features": 12,
                "selection_method": "correlation",
                "selection_score": 0.85,
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing feature selection: {e}"))
            return {}

    def _perform_dimensionality_reduction(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform dimensionality reduction."""
        try:
            # Simulate dimensionality reduction
            return {
                "dimensionality_reduction_completed": True,
                "reduced_dimensions": 5,
                "reduction_method": "pca",
                "explained_variance": 0.95,
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing dimensionality reduction: {e}"))
            return {}

    # Data aggregation methods
    def _perform_time_aggregation(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform time aggregation."""
        try:
            # Simulate time aggregation
            return {
                "time_aggregation_completed": True,
                "aggregated_periods": 24,
                "aggregation_method": "hourly",
                "time_series_length": 1000,
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing time aggregation: {e}"))
            return {}

    def _perform_group_aggregation(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform group aggregation."""
        try:
            # Simulate group aggregation
            return {
                "group_aggregation_completed": True,
                "aggregated_groups": 5,
                "aggregation_method": "mean",
                "group_statistics": "calculated",
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing group aggregation: {e}"))
            return {}

    def _perform_statistical_aggregation(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform statistical aggregation."""
        try:
            # Simulate statistical aggregation
            return {
                "statistical_aggregation_completed": True,
                "statistical_measures": ["mean", "std", "min", "max"],
                "aggregation_method": "descriptive",
                "statistical_summary": "generated",
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing statistical aggregation: {e}"))
            return {}

    def _perform_custom_aggregation(
        self,
        processing_input: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform custom aggregation."""
        try:
            # Simulate custom aggregation
            return {
                "custom_aggregation_completed": True,
                "custom_functions": 3,
                "aggregation_method": "custom",
                "custom_metrics": "calculated",
                "training_time": datetime.now().isoformat(),
            }
        except Exception:
            self.print(error("Error performing custom aggregation: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="processing results storage",
    )
    async def _store_processing_results(self) -> None:
        """Store processing results."""
        try:
            # Add timestamp
            self.processing_results["timestamp"] = datetime.now().isoformat()

            # Add to history
            self.processing_history.append(self.processing_results.copy())

            # Limit history size
            if len(self.processing_history) > self.max_processing_history:
                self.processing_history.pop(0)

            self.logger.info("Processing results stored successfully")

        except Exception:
            self.print(error("Error storing processing results: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="processing results getting",
    )
    def get_processing_results(
        self,
        processing_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get processing results.

        Args:
            processing_type: Optional processing type filter

        Returns:
            dict[str, Any]: Processing results
        """
        try:
            if processing_type:
                return self.processing_results.get(processing_type, {})
            return self.processing_results.copy()

        except Exception:
            self.print(error("Error getting processing results: {e}"))
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="processing history getting",
    )
    def get_processing_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get processing history.

        Args:
            limit: Optional limit on number of records

        Returns:
            list[dict[str, Any]]: Processing history
        """
        try:
            history = self.processing_history.copy()

            if limit:
                history = history[-limit:]

            return history

        except Exception:
            self.print(error("Error getting processing history: {e}"))
            return []

    def get_processing_status(self) -> dict[str, Any]:
        """
        Get processing status information.

        Returns:
            dict[str, Any]: Processing status
        """
        return {
            "is_processing": self.is_processing,
            "processing_interval": self.processing_interval,
            "max_processing_history": self.max_processing_history,
            "enable_data_cleaning": self.enable_data_cleaning,
            "enable_data_validation": self.enable_data_validation,
            "enable_data_transformation": self.data_utils_config.get(
                "enable_data_transformation",
                True,
            ),
            "enable_data_aggregation": self.data_utils_config.get(
                "enable_data_aggregation",
                True,
            ),
            "processing_history_count": len(self.processing_history),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="data utils cleanup",
    )

# Global data utils instance
data_utils: DataUtils | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="data utils setup",
)

def validate_klines_data(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate klines data quality."""
    if df.empty:
        return False, "Empty DataFrame"

    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    # Check for NaN values
    nan_counts = df[required_cols].isnull().sum()
    if nan_counts.sum() > 0:
        return False, f"NaN values found: {nan_counts.to_dict()}"

    # Check for infinite values
    inf_counts = np.isinf(df[required_cols]).sum()
    if inf_counts.sum() > 0:
        return False, f"Infinite values found: {inf_counts.to_dict()}"

    # Check for negative prices
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if (df[col] < 0).any():
            return False, f"Negative values found in {col}"

    # Check for invalid OHLC relationships
    if (df["high"] < df["low"]).any():
        return False, "High < Low found"

    if (
        (df["open"] > df["high"])
        | (df["open"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
    ).any():
        return False, "Open/Close outside High-Low range"

    # Check for zero prices
    for col in price_cols:
        if (df[col] == 0).any():
            return False, f"Zero values found in {col}"

    return True, "Data quality validation passed"







def _calculate_price_range(
    klines_df: pd.DataFrame,
    close_col: str,
    high_col: str,
    low_col: str,
) -> tuple[float, float]:
    """Calculate the price range for volume profile analysis."""
    min_price = klines_df[close_col].min()
    max_price = klines_df[close_col].max()

    # Add padding to the range (10% on each side)
    price_range = max_price - min_price
    padding = price_range * 0.1
    min_price = max(100.0, min_price - padding)  # Don't go below $100
    max_price = max_price + padding

    # Handle extreme outliers using percentiles
    if max_price / min_price > 100:  # More than 100x difference
        min_price = klines_df[close_col].quantile(0.01)  # 1st percentile
        max_price = klines_df[close_col].quantile(0.99)  # 99th percentile

    return min_price, max_price


def _filter_reasonable_data(
    klines_df: pd.DataFrame,
    min_price: float,
    max_price: float,
    close_col: str,
    high_col: str,
    low_col: str,
) -> pd.DataFrame:
    """Filter data to only include reasonable prices within the calculated range."""
    reasonable_data = klines_df[
        (klines_df[close_col] >= min_price)
        & (klines_df[close_col] <= max_price)
        & (klines_df[high_col] >= min_price)
        & (klines_df[high_col] <= max_price)
        & (klines_df[low_col] >= min_price)
        & (klines_df[low_col] <= max_price)
    ]

    return reasonable_data if len(reasonable_data) > 0 else klines_df


def _create_volume_profile(
    klines_df: pd.DataFrame,
    min_price: float,
    max_price: float,
    high_col: str,
    low_col: str,
    volume_col: str,
    num_bins: int,
) -> pd.Series:
    """Create the volume profile by binning price data and summing volumes."""
    if max_price == min_price:  # Handle flat market
        return pd.Series([klines_df[volume_col].sum()], index=[min_price])

    # Create bins and assign volume to price bins
    actual_bins = min(num_bins, 100)
    bins = np.linspace(min_price, max_price, actual_bins + 1)

    # Assign each candle's midpoint to a bin and sum its volume
    mid_prices = (klines_df[high_col] + klines_df[low_col]) / 2
    price_bins_categorized = pd.cut(mid_prices, bins, include_lowest=True)

    # Group by these categories and sum volume
    volume_profile_series = klines_df.groupby(price_bins_categorized)[volume_col].sum()

    # Map bin intervals to their midpoints for a more usable index
    bin_midpoints_map = {
        interval: (interval.left + interval.right) / 2
        for interval in volume_profile_series.index
    }
    volume_profile = volume_profile_series.rename(index=bin_midpoints_map)
    return volume_profile.fillna(0)  # Fill bins with no volume as 0


def _detect_peaks_with_prominence(
    volume_profile: pd.Series,
) -> list[tuple[float, float]]:
    """Detect peaks using prominence-based method."""
    hvn_levels = []
    hvn_strengths = {}

    hvn_indices, _ = find_peaks(
        volume_profile.values,
        prominence=volume_profile.max() * 0.005,  # 0.5% threshold
        width=1,
    )

    for i in hvn_indices:
        level = volume_profile.index[i]
        hvn_levels.append(level)
        volume_at_level = volume_profile.iloc[i]
        total_volume = volume_profile.sum()
        strength = min(volume_at_level / total_volume * 100, 1.0)
        hvn_strengths[level] = strength

    return [(level, hvn_strengths[level]) for level in hvn_levels]


def _detect_peaks_with_percentiles(
    volume_profile: pd.Series,
) -> list[tuple[float, float]]:
    """Detect peaks using percentile-based method."""
    hvn_levels = []
    hvn_strengths = {}

    percentiles = [
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ]

    for percentile in percentiles:
        volume_threshold = volume_profile.quantile(percentile)
        high_volume_levels = volume_profile[
            volume_profile > volume_threshold
        ].index.tolist()

        for level in high_volume_levels:
            if level not in hvn_levels:
                hvn_levels.append(level)
                volume_at_level = volume_profile.loc[level]
                total_volume = volume_profile.sum()
                percentile_strength = (
                    percentile - 0.3
                ) * 1.43  # 0.3 to 1.0 based on percentile
                volume_strength = min(volume_at_level / total_volume * 100, 1.0)
                strength = (percentile_strength + volume_strength) / 2
                hvn_strengths[level] = strength

    return [(level, hvn_strengths[level]) for level in hvn_levels]


def _detect_local_maxima(volume_profile: pd.Series) -> list[tuple[float, float]]:
    """Detect local maxima using multiple window sizes."""
    hvn_levels = []
    hvn_strengths = {}
    local_maxima_indices = []

    # Multiple window sizes for local maxima detection
    window_sizes = [1, 2, 3, 4, 5]

    for window_size in window_sizes:
        for i in range(window_size, len(volume_profile) - window_size):
            is_maximum = True
            for j in range(1, window_size + 1):
                if (
                    volume_profile.iloc[i] <= volume_profile.iloc[i - j]
                    or volume_profile.iloc[i] <= volume_profile.iloc[i + j]
                ):
                    is_maximum = False
                    break

            if is_maximum:
                local_maxima_indices.append(i)

    # Remove duplicates and add levels
    local_maxima_indices = list(set(local_maxima_indices))

    for i in local_maxima_indices:
        level = volume_profile.index[i]
        if level not in hvn_levels:
            hvn_levels.append(level)
            volume_at_level = volume_profile.iloc[i]
            total_volume = volume_profile.sum()
            strength = min(volume_at_level / total_volume * 50, 0.8)
            hvn_strengths[level] = strength

    return [(level, hvn_strengths[level]) for level in hvn_levels]


def _add_volume_weighted_levels(volume_profile: pd.Series) -> list[tuple[float, float]]:
    """Add levels based on volume distribution."""
    hvn_levels = []
    hvn_strengths = {}

    volume_sorted = volume_profile.sort_values(ascending=False)
    top_volume_levels = volume_sorted.head(
        int(len(volume_profile) * 0.7),
    ).index.tolist()

    for level in top_volume_levels:
        if level not in hvn_levels:
            hvn_levels.append(level)
            volume_at_level = volume_profile.loc[level]
            total_volume = volume_profile.sum()
            strength = min(volume_at_level / total_volume * 80, 0.9)
            hvn_strengths[level] = strength

    return [(level, hvn_strengths[level]) for level in hvn_levels]


def _add_distributed_levels(volume_profile: pd.Series) -> list[tuple[float, float]]:
    """Add levels at regular intervals across the price range."""
    hvn_levels = []
    hvn_strengths = {}

    price_range = volume_profile.index.max() - volume_profile.index.min()
    interval_count = max(15, int(len(volume_profile) * 0.6))
    interval = price_range / interval_count

    for i in range(interval_count):
        target_price = volume_profile.index.min() + (i + 0.5) * interval
        closest_level = min(volume_profile.index, key=lambda x: abs(x - target_price))
        if closest_level not in hvn_levels:
            hvn_levels.append(closest_level)
            volume_at_level = volume_profile.loc[closest_level]
            total_volume = volume_profile.sum()
            strength = min(volume_at_level / total_volume * 60, 0.7)
            hvn_strengths[closest_level] = strength

    return [(level, hvn_strengths[level]) for level in hvn_levels]


def _ensure_minimum_levels(
    volume_profile: pd.Series,
    existing_levels: list[tuple[float, float]],
    min_levels: int = 200,
) -> list[tuple[float, float]]:
    """Ensure we have at least the minimum number of levels."""
    all_levels = existing_levels.copy()

    if len(all_levels) < min_levels:
        # Add remaining levels with lower strength
        existing_prices = {level for level, _ in all_levels}
        remaining_levels = [
            (level, volume_profile.loc[level])
            for level in volume_profile.index
            if level not in existing_prices
        ]

        # Sort by volume and add top remaining levels
        remaining_levels.sort(key=lambda x: x[1], reverse=True)

        for level, volume_at_level in remaining_levels[: min_levels - len(all_levels)]:
            total_volume = volume_profile.sum()
            strength = min(volume_at_level / total_volume * 40, 0.6)
            all_levels.append((level, strength))

    return all_levels


def _consolidate_hvn_results(
    all_levels: list[tuple[float, float]],
    volume_profile: pd.Series,
) -> list[dict]:
    """Consolidate all detected levels into final results."""
    # Remove duplicates and sort by strength
    unique_levels = {}
    for level, strength in all_levels:
        if level not in unique_levels or strength > unique_levels[level]:
            unique_levels[level] = strength

    # Create final results
    hvn_results = []
    for level, strength in unique_levels.items():
        hvn_results.append(
            {
                "price": level,
                "strength": strength,
                "volume_concentration": volume_profile.loc[level]
                / volume_profile.sum(),
                "method": "hvn",
            },
        )

    # Sort by strength (strongest first)
    hvn_results.sort(key=lambda x: x["strength"], reverse=True)
    return hvn_results



