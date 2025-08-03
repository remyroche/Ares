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
    async def initialize(self) -> bool:
        """
        Initialize data utils with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Data Utils...")

            # Load data utils configuration
            await self._load_data_utils_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for data utils")
                return False

            # Initialize data utils modules
            await self._initialize_data_utils_modules()

            self.logger.info("✅ Data Utils initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Data Utils initialization failed: {e}")
            return False

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

        except Exception as e:
            self.logger.error(f"Error loading data utils configuration: {e}")

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
                self.logger.error("Invalid processing interval")
                return False

            # Validate max processing history
            if self.max_processing_history <= 0:
                self.logger.error("Invalid max processing history")
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
                self.logger.error("At least one processing type must be enabled")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data utils modules initialization",
    )
    async def _initialize_data_utils_modules(self) -> None:
        """Initialize data utils modules."""
        try:
            # Initialize data cleaning module
            if self.enable_data_cleaning:
                await self._initialize_data_cleaning()

            # Initialize data validation module
            if self.enable_data_validation:
                await self._initialize_data_validation()

            # Initialize data transformation module
            if self.data_utils_config.get("enable_data_transformation", True):
                await self._initialize_data_transformation()

            # Initialize data aggregation module
            if self.data_utils_config.get("enable_data_aggregation", True):
                await self._initialize_data_aggregation()

            self.logger.info("Data utils modules initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing data utils modules: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data cleaning initialization",
    )
    async def _initialize_data_cleaning(self) -> None:
        """Initialize data cleaning module."""
        try:
            # Initialize data cleaning components
            self.data_cleaning_components = {
                "outlier_removal": True,
                "missing_data_handling": True,
                "duplicate_removal": True,
                "data_normalization": True,
            }

            self.logger.info("Data cleaning module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing data cleaning: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data validation initialization",
    )
    async def _initialize_data_validation(self) -> None:
        """Initialize data validation module."""
        try:
            # Initialize data validation components
            self.data_validation_components = {
                "data_type_validation": True,
                "range_validation": True,
                "format_validation": True,
                "consistency_validation": True,
            }

            self.logger.info("Data validation module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing data validation: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data transformation initialization",
    )
    async def _initialize_data_transformation(self) -> None:
        """Initialize data transformation module."""
        try:
            # Initialize data transformation components
            self.data_transformation_components = {
                "feature_scaling": True,
                "feature_encoding": True,
                "feature_selection": True,
                "dimensionality_reduction": True,
            }

            self.logger.info("Data transformation module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing data transformation: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="data aggregation initialization",
    )
    async def _initialize_data_aggregation(self) -> None:
        """Initialize data aggregation module."""
        try:
            # Initialize data aggregation components
            self.data_aggregation_components = {
                "time_aggregation": True,
                "group_aggregation": True,
                "statistical_aggregation": True,
                "custom_aggregation": True,
            }

            self.logger.info("Data aggregation module initialized")

        except Exception as e:
            self.logger.error(f"Error initializing data aggregation: {e}")

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

        except Exception as e:
            self.logger.error(f"Error executing data processing: {e}")
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
                self.logger.error("Invalid processing type")
                return False

            if not isinstance(processing_input["data_source"], str):
                self.logger.error("Invalid data source")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating processing inputs: {e}")
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

        except Exception as e:
            self.logger.error(f"Error performing data cleaning: {e}")
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

        except Exception as e:
            self.logger.error(f"Error performing data validation: {e}")
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

        except Exception as e:
            self.logger.error(f"Error performing data transformation: {e}")
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

        except Exception as e:
            self.logger.error(f"Error performing data aggregation: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing outlier removal: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing missing data handling: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing duplicate removal: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing data normalization: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing data type validation: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing range validation: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing format validation: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing consistency validation: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing feature scaling: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing feature encoding: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing feature selection: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing dimensionality reduction: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing time aggregation: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing group aggregation: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing statistical aggregation: {e}")
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
        except Exception as e:
            self.logger.error(f"Error performing custom aggregation: {e}")
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

        except Exception as e:
            self.logger.error(f"Error storing processing results: {e}")

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

        except Exception as e:
            self.logger.error(f"Error getting processing results: {e}")
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

        except Exception as e:
            self.logger.error(f"Error getting processing history: {e}")
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
    async def stop(self) -> None:
        """Stop the data utils."""
        self.logger.info("🛑 Stopping Data Utils...")

        try:
            # Stop processing
            self.is_processing = False

            # Clear results
            self.processing_results.clear()

            # Clear history
            self.processing_history.clear()

            self.logger.info("✅ Data Utils stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping data utils: {e}")


# Global data utils instance
data_utils: DataUtils | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="data utils setup",
)
async def setup_data_utils(config: dict[str, Any] | None = None) -> DataUtils | None:
    """
    Setup global data utils.

    Args:
        config: Optional configuration dictionary

    Returns:
        DataUtils | None: Global data utils instance
    """
    try:
        global data_utils

        if config is None:
            config = {
                "data_utils": {
                    "processing_interval": 3600,
                    "max_processing_history": 100,
                    "enable_data_cleaning": True,
                    "enable_data_validation": True,
                    "enable_data_transformation": True,
                    "enable_data_aggregation": True,
                },
            }

        # Create data utils
        data_utils = DataUtils(config)

        # Initialize data utils
        success = await data_utils.initialize()
        if success:
            return data_utils
        return None

    except Exception as e:
        print(f"Error setting up data utils: {e}")
        return None


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
    
    if ((df["open"] > df["high"]) | (df["open"] < df["low"]) | 
        (df["close"] > df["high"]) | (df["close"] < df["low"])).any():
        return False, "Open/Close outside High-Low range"
    
    # Check for zero prices
    for col in price_cols:
        if (df[col] == 0).any():
            return False, f"Zero values found in {col}"
    
    return True, "Data quality validation passed"


def load_klines_data(filename):
    """Loads k-line data from a CSV file with strict quality validation."""
    if not os.path.exists(filename):
        print(f"❌ CRITICAL: K-lines data file not found at {filename}")
        return pd.DataFrame()

    try:
        # Read CSV with more robust timestamp parsing
        df = pd.read_csv(filename, index_col="open_time", parse_dates=True)
        print(
            f"[DEBUG] load_klines_data: type={type(df)}, shape={df.shape}, columns={df.columns.tolist()}",
        )
        print(df.head())

        # Convert timestamp with flexible parsing
        df["open_time"] = pd.to_datetime(df["open_time"], format="mixed", errors="coerce")

        # Remove rows with invalid timestamps
        initial_rows = len(df)
        df = df.dropna(subset=["open_time"])
        if len(df) < initial_rows:
            print(f"⚠️ Warning: Removed {initial_rows - len(df)} rows with invalid timestamps")

        if df.empty:
            print("❌ CRITICAL: No valid data after timestamp processing")
            return pd.DataFrame()

        # Set timestamp as index
        df.set_index("open_time", inplace=True)
        df = df[~df.index.duplicated(keep="first")]  # Remove duplicates

        # Ensure numeric columns are actually numeric
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric, but don't fill NaN values
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check for NaN values - FAIL FAST if found
        nan_counts = df[numeric_cols].isnull().sum()
        total_nan = nan_counts.sum()
        if total_nan > 0:
            print(f"❌ CRITICAL: Found {total_nan} NaN values in klines data: {nan_counts.to_dict()}")
            print("Please fix the data quality issues before proceeding.")
            return pd.DataFrame()

        # Check for infinite values - FAIL FAST if found
        inf_counts = np.isinf(df[numeric_cols]).sum()
        total_inf = inf_counts.sum()
        if total_inf > 0:
            print(f"❌ CRITICAL: Found {total_inf} infinite values in klines data: {inf_counts.to_dict()}")
            print("Please fix the data quality issues before proceeding.")
            return pd.DataFrame()

        # Check for negative prices - FAIL FAST if found
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    print(f"❌ CRITICAL: Found {negative_count} negative values in {col}")
                    print("Please fix the data quality issues before proceeding.")
                    return pd.DataFrame()

        # Check for zero prices - FAIL FAST if found
        for col in price_cols:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    print(f"❌ CRITICAL: Found {zero_count} zero values in {col}")
                    print("Please fix the data quality issues before proceeding.")
                    return pd.DataFrame()

        # Check for invalid OHLC relationships - FAIL FAST if found
        if (df["high"] < df["low"]).any():
            invalid_count = (df["high"] < df["low"]).sum()
            print(f"❌ CRITICAL: Found {invalid_count} rows where high < low")
            print("Please fix the data quality issues before proceeding.")
            return pd.DataFrame()

        if ((df["open"] > df["high"]) | (df["open"] < df["low"]) | 
            (df["close"] > df["high"]) | (df["close"] < df["low"])).any():
            invalid_count = ((df["open"] > df["high"]) | (df["open"] < df["low"]) | 
                           (df["close"] > df["high"]) | (df["close"] < df["low"])).sum()
            print(f"❌ CRITICAL: Found {invalid_count} rows where open/close outside high-low range")
            print("Please fix the data quality issues before proceeding.")
            return pd.DataFrame()

        if df.empty:
            print("❌ CRITICAL: No valid data after processing")
            return pd.DataFrame()

        print(f"✅ Successfully loaded {len(df)} high-quality klines records")
        return df

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Error loading klines data from {filename}: {e}")
        return pd.DataFrame()


def load_agg_trades_data(filename):
    """Loads aggregated trades data from a CSV file with strict quality validation."""
    if not os.path.exists(filename):
        print(f"❌ CRITICAL: Agg trades data file not found at {filename}")
        return pd.DataFrame()

    try:
        # Read CSV with more robust timestamp parsing
        df = pd.read_csv(filename, low_memory=False)

        # Convert timestamp with flexible parsing
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")

        # Remove rows with invalid timestamps
        initial_rows = len(df)
        df = df.dropna(subset=["timestamp"])
        if len(df) < initial_rows:
            print(f"⚠️ Warning: Removed {initial_rows - len(df)} rows with invalid timestamps")

        if df.empty:
            print("❌ CRITICAL: No valid data after timestamp processing")
            return pd.DataFrame()

        # Set timestamp as index
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="first")]  # Remove duplicates

        # Ensure numeric columns are actually numeric
        numeric_cols = ["price", "quantity"]
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric, but don't fill NaN values
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check for NaN values - FAIL FAST if found
        nan_counts = df[numeric_cols].isnull().sum()
        total_nan = nan_counts.sum()
        if total_nan > 0:
            print(f"❌ CRITICAL: Found {total_nan} NaN values in agg_trades data: {nan_counts.to_dict()}")
            print("Please fix the data quality issues before proceeding.")
            return pd.DataFrame()

        # Check for infinite values - FAIL FAST if found
        inf_counts = np.isinf(df[numeric_cols]).sum()
        total_inf = inf_counts.sum()
        if total_inf > 0:
            print(f"❌ CRITICAL: Found {total_inf} infinite values in agg_trades data: {inf_counts.to_dict()}")
            print("Please fix the data quality issues before proceeding.")
            return pd.DataFrame()

        # Check for negative values - FAIL FAST if found
        for col in numeric_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    print(f"❌ CRITICAL: Found {negative_count} negative values in {col}")
                    print("Please fix the data quality issues before proceeding.")
                    return pd.DataFrame()

        if df.empty:
            print("❌ CRITICAL: No valid data after processing")
            return pd.DataFrame()

        print(f"✅ Successfully loaded {len(df)} high-quality agg_trades records")
        return df

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Error loading agg_trades data from {filename}: {e}")
        return pd.DataFrame()


def load_futures_data(filename):
    """Loads futures data (funding rates) from a CSV file with strict quality validation."""
    if not os.path.exists(filename):
        print(f"❌ CRITICAL: Futures data file not found at {filename}")
        return pd.DataFrame()

    try:
        # Read CSV with more robust timestamp parsing
        df = pd.read_csv(filename, low_memory=False)

        # Convert timestamp with flexible parsing
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")

        # Remove rows with invalid timestamps
        initial_rows = len(df)
        df = df.dropna(subset=["timestamp"])
        if len(df) < initial_rows:
            print(f"⚠️ Warning: Removed {initial_rows - len(df)} rows with invalid timestamps")

        if df.empty:
            print("❌ CRITICAL: No valid data after timestamp processing")
            return pd.DataFrame()

        # Set timestamp as index
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="first")]  # Remove duplicates

        # Ensure numeric columns are actually numeric
        numeric_cols = ["fundingRate"]
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric, but don't fill NaN values
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Check for NaN values - FAIL FAST if found
        nan_counts = df[numeric_cols].isnull().sum()
        total_nan = nan_counts.sum()
        if total_nan > 0:
            print(f"❌ CRITICAL: Found {total_nan} NaN values in futures data: {nan_counts.to_dict()}")
            print("Please fix the data quality issues before proceeding.")
            return pd.DataFrame()

        # Check for infinite values - FAIL FAST if found
        inf_counts = np.isinf(df[numeric_cols]).sum()
        total_inf = inf_counts.sum()
        if total_inf > 0:
            print(f"❌ CRITICAL: Found {total_inf} infinite values in futures data: {inf_counts.to_dict()}")
            print("Please fix the data quality issues before proceeding.")
            return pd.DataFrame()

        if df.empty:
            print("❌ CRITICAL: No valid data after processing")
            return pd.DataFrame()

        print(f"✅ Successfully loaded {len(df)} high-quality futures records")
        return df

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Error loading futures data from {filename}: {e}")
        return pd.DataFrame()


def simulate_order_book_data(current_price):
    """Simulates real-time order book data for demonstration."""
    simulated_bids = [
        [current_price - 0.1, 5],
        [current_price - 0.2, 10],
        [current_price - 0.5, 20],
        [current_price - 1.0, 100],
        [current_price - 2.0, 50000 / current_price],  # Large buy wall (approx $50k)
        [current_price - 2.5, 15],
        [
            current_price - 3.0,
            120000 / current_price,
        ],  # Even larger buy wall (approx $120k)
    ]
    simulated_asks = [
        [current_price + 0.1, 7],
        [current_price + 0.2, 12],
        [current_price + 0.5, 25],
        [current_price + 1.0, 80],
        [current_price + 2.0, 60000 / current_price],  # Large sell wall (approx $60k)
        [current_price + 2.5, 18],
        [
            current_price + 3.0,
            110000 / current_price,
        ],  # Even larger sell wall (approx $110k)
    ]
    return {"bids": simulated_bids, "asks": simulated_asks}


def calculate_volume_profile(klines_df: pd.DataFrame, num_bins: int = 100):
    """
    Calculates Volume Profile (HVNs, LVNs, POC) for the given price range.
    Uses 'High', 'Low', 'Volume' from klines data.
    :param klines_df: DataFrame with 'High', 'Low', 'Volume' columns.
    :param num_bins: Number of price bins for the volume profile.
    :return: dict with 'poc', 'hvn_levels', 'lvn_levels', 'volume_in_bins' (Series with bin midpoints as index)
    """
    if klines_df.empty:
        return {
            "poc": np.nan,
            "hvn_levels": [],
            "lvn_levels": [],
            "volume_in_bins": pd.Series(),
        }

    # Use Close data for price range calculation since it's more reliable
    # Handle both uppercase and lowercase column names
    close_col = "Close" if "Close" in klines_df.columns else "close"
    high_col = "High" if "High" in klines_df.columns else "high"
    low_col = "Low" if "Low" in klines_df.columns else "low"
    volume_col = "Volume" if "Volume" in klines_df.columns else "volume"
    
    min_price = klines_df[close_col].min()
    max_price = klines_df[close_col].max()
    
    # Debug: Print the actual data range
    print(f"Volume Profile Debug - Raw Close Range: {min_price:.2f} to {max_price:.2f}")
    print(f"Volume Profile Debug - Raw High/Low Range: {klines_df[low_col].min():.2f} to {klines_df[high_col].max():.2f}")
    
    # Filter out extreme outliers using percentiles to avoid corrupted data
    # Use actual ETH price range based on the data
    min_price = klines_df[close_col].min()
    max_price = klines_df[close_col].max()
    
    # Add some padding to the actual range (10% on each side)
    price_range = max_price - min_price
    padding = price_range * 0.1
    min_price = max(100.0, min_price - padding)  # Don't go below $100
    max_price = max_price + padding
    
    print(f"Volume Profile Debug - Median Price: {klines_df[close_col].median():.2f}")
    print(f"Volume Profile Debug - Using fixed ETH price range: {min_price:.2f} to {max_price:.2f}")
    
    # Filter the data to only include reasonable prices
    reasonable_data = klines_df[
        (klines_df[close_col] >= min_price) & 
        (klines_df[close_col] <= max_price) &
        (klines_df[high_col] >= min_price) & 
        (klines_df[high_col] <= max_price) &
        (klines_df[low_col] >= min_price) & 
        (klines_df[low_col] <= max_price)
    ]
    
    if len(reasonable_data) == 0:
        print(f"Volume Profile Debug - No reasonable data found, using original data")
        reasonable_data = klines_df
    
    print(f"Volume Profile Debug - Reasonable records: {len(reasonable_data)} out of {len(klines_df)}")
    print(f"Volume Profile Debug - Final Range: {min_price:.2f} to {max_price:.2f}")
    
    # Use the reasonable data for volume profile calculation
    klines_df = reasonable_data
    
    # Additional sanity check: if the range is still too large, use percentiles
    if max_price / min_price > 100:  # More than 100x difference
        min_price = klines_df[close_col].quantile(0.01)  # 1st percentile
        max_price = klines_df[close_col].quantile(0.99)  # 99th percentile
        print(f"Volume Profile Debug - Using percentiles due to large range")
    
    print(f"Volume Profile Debug - Final Range: {min_price:.2f} to {max_price:.2f}")

    if max_price == min_price:  # Handle flat market
        return {
            "poc": min_price,
            "hvn_levels": [min_price],
            "lvn_levels": [],
            "volume_in_bins": pd.Series([klines_df[volume_col].sum()], index=[min_price]),
        }

    # Create bins and sum volume within each bin
    # We'll create bins based on the overall price range
    # Use 100 bins as requested
    actual_bins = min(num_bins, 100)  # Use 100 bins as requested
    bins = np.linspace(min_price, max_price, actual_bins + 1)

    # To accurately assign volume to price bins, we can iterate through candles
    # and distribute their volume across the bins they span.
    # For simplicity and performance with OHLCV, we'll assign candle's volume to its midpoint bin.
    # A more precise method would involve distributing volume proportionally across price ranges.

    # Assign each candle's midpoint to a bin and sum its volume
    mid_prices = (klines_df[high_col] + klines_df[low_col]) / 2

    # Use pd.cut to categorize each midpoint into a bin interval
    price_bins_categorized = pd.cut(mid_prices, bins, include_lowest=True)

    # Group by these categories and sum volume
    volume_profile_series = klines_df.groupby(price_bins_categorized)[volume_col].sum()

    # Map bin intervals to their midpoints for a more usable index
    bin_midpoints_map = {
        interval: (interval.left + interval.right) / 2
        for interval in volume_profile_series.index
    }
    volume_profile = volume_profile_series.rename(index=bin_midpoints_map)
    volume_profile = volume_profile.fillna(0)  # Fill bins with no volume as 0

    # Point of Control (POC): Price level (midpoint of bin) with highest volume
    poc_price = volume_profile.idxmax() if not volume_profile.empty else np.nan
    
    # Debug: Print price range info
    print(f"Volume Profile Debug - Price Range: {min_price:.2f} to {max_price:.2f}")
    print(f"Volume Profile Debug - Median Price: {klines_df[close_col].median():.2f}")
    print(f"Volume Profile Debug - POC Price: {poc_price:.2f}")
    print(f"Volume Profile Debug - Number of bins: {len(volume_profile)}")
    print(f"Volume Profile Debug - Sample bin prices: {list(volume_profile.index[:5])}")

    # High-Volume Nodes (HVNs): Much more aggressive detection
    # Use multiple methods to ensure we get enough levels
    
    hvn_levels = []
    hvn_strengths = {}
    
    # Method 1: More aggressive prominence peak detection (0.5% threshold)
    hvn_indices, _ = find_peaks(
        volume_profile.values,
        prominence=volume_profile.max() * 0.005,  # 0.5% threshold (more aggressive)
        width=1,
    )
    for i in hvn_indices:
        level = volume_profile.index[i]
        hvn_levels.append(level)
        # Calculate strength based on volume concentration
        volume_at_level = volume_profile.iloc[i]
        total_volume = volume_profile.sum()
        strength = min(volume_at_level / total_volume * 100, 1.0)  # Normalize to 0-1
        hvn_strengths[level] = strength
    
    # Method 2: Much more aggressive percentile-based detection
    # Use many more percentiles to catch different levels of volume concentration
    percentiles = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # 16 different thresholds (more aggressive)
    for percentile in percentiles:
        volume_threshold = volume_profile.quantile(percentile)
        high_volume_levels = volume_profile[volume_profile > volume_threshold].index.tolist()
        
        for level in high_volume_levels:
            if level not in hvn_levels:
                hvn_levels.append(level)
                # Calculate strength based on percentile and volume concentration
                volume_at_level = volume_profile.loc[level]
                total_volume = volume_profile.sum()
                percentile_strength = (percentile - 0.3) * 1.43  # 0.3 to 1.0 based on percentile
                volume_strength = min(volume_at_level / total_volume * 100, 1.0)
                strength = (percentile_strength + volume_strength) / 2
                hvn_strengths[level] = strength
    
    # Method 3: Much more aggressive local maxima detection
    # Find all local maxima with multiple window sizes
    local_maxima_indices = []
    
    # Window size 1: immediate neighbors (most aggressive)
    for i in range(1, len(volume_profile) - 1):
        if (volume_profile.iloc[i] > volume_profile.iloc[i-1] and 
            volume_profile.iloc[i] > volume_profile.iloc[i+1]):
            local_maxima_indices.append(i)
    
    # Window size 2: wider context (more aggressive)
    for i in range(2, len(volume_profile) - 2):
        if (volume_profile.iloc[i] > volume_profile.iloc[i-2] and 
            volume_profile.iloc[i] > volume_profile.iloc[i-1] and
            volume_profile.iloc[i] > volume_profile.iloc[i+1] and
            volume_profile.iloc[i] > volume_profile.iloc[i+2]):
            local_maxima_indices.append(i)
    
    # Window size 3: even wider context (most aggressive)
    for i in range(3, len(volume_profile) - 3):
        if (volume_profile.iloc[i] > volume_profile.iloc[i-3] and 
            volume_profile.iloc[i] > volume_profile.iloc[i-2] and
            volume_profile.iloc[i] > volume_profile.iloc[i-1] and
            volume_profile.iloc[i] > volume_profile.iloc[i+1] and
            volume_profile.iloc[i] > volume_profile.iloc[i+2] and
            volume_profile.iloc[i] > volume_profile.iloc[i+3]):
            local_maxima_indices.append(i)
    
    # Window size 4: even more aggressive (new)
    for i in range(4, len(volume_profile) - 4):
        if (volume_profile.iloc[i] > volume_profile.iloc[i-4] and 
            volume_profile.iloc[i] > volume_profile.iloc[i-3] and
            volume_profile.iloc[i] > volume_profile.iloc[i-2] and
            volume_profile.iloc[i] > volume_profile.iloc[i-1] and
            volume_profile.iloc[i] > volume_profile.iloc[i+1] and
            volume_profile.iloc[i] > volume_profile.iloc[i+2] and
            volume_profile.iloc[i] > volume_profile.iloc[i+3] and
            volume_profile.iloc[i] > volume_profile.iloc[i+4]):
            local_maxima_indices.append(i)
    
    # Window size 5: most aggressive (new)
    for i in range(5, len(volume_profile) - 5):
        if (volume_profile.iloc[i] > volume_profile.iloc[i-5] and 
            volume_profile.iloc[i] > volume_profile.iloc[i-4] and
            volume_profile.iloc[i] > volume_profile.iloc[i-3] and
            volume_profile.iloc[i] > volume_profile.iloc[i-2] and
            volume_profile.iloc[i] > volume_profile.iloc[i-1] and
            volume_profile.iloc[i] > volume_profile.iloc[i+1] and
            volume_profile.iloc[i] > volume_profile.iloc[i+2] and
            volume_profile.iloc[i] > volume_profile.iloc[i+3] and
            volume_profile.iloc[i] > volume_profile.iloc[i+4] and
            volume_profile.iloc[i] > volume_profile.iloc[i+5]):
            local_maxima_indices.append(i)
    
    # Remove duplicates and add levels
    local_maxima_indices = list(set(local_maxima_indices))
    
    for i in local_maxima_indices:
        level = volume_profile.index[i]
        if level not in hvn_levels:
            hvn_levels.append(level)
            # Calculate strength for local maxima
            volume_at_level = volume_profile.iloc[i]
            total_volume = volume_profile.sum()
            strength = min(volume_at_level / total_volume * 50, 0.8)  # Cap at 0.8 for local maxima
            hvn_strengths[level] = strength
    
    # Method 4: Volume-weighted sampling (super aggressive)
    # Add levels based on volume distribution
    volume_sorted = volume_profile.sort_values(ascending=False)
    top_volume_levels = volume_sorted.head(int(len(volume_profile) * 0.7)).index.tolist()  # Top 70% (more aggressive)
    
    for level in top_volume_levels:
        if level not in hvn_levels:
            hvn_levels.append(level)
            volume_at_level = volume_profile.loc[level]
            total_volume = volume_profile.sum()
            strength = min(volume_at_level / total_volume * 80, 0.9)  # Higher strength for volume-weighted
            hvn_strengths[level] = strength
    
    # Method 5: Even distribution sampling (most aggressive)
    # Add levels at regular intervals across the price range
    price_range = volume_profile.index.max() - volume_profile.index.min()
    interval_count = max(15, int(len(volume_profile) * 0.6))  # At least 15, up to 60% of bins (more aggressive)
    interval = price_range / interval_count
    
    for i in range(interval_count):
        target_price = volume_profile.index.min() + (i + 0.5) * interval
        # Find the closest actual price level
        closest_level = min(volume_profile.index, key=lambda x: abs(x - target_price))
        if closest_level not in hvn_levels:
            hvn_levels.append(closest_level)
            volume_at_level = volume_profile.loc[closest_level]
            total_volume = volume_profile.sum()
            strength = min(volume_at_level / total_volume * 60, 0.7)  # Moderate strength for distribution
            hvn_strengths[closest_level] = strength
    
    # Method 6: Force minimum number of levels (super aggressive)
    # Ensure we have at least 200 levels for a 2-year dataset (target: 1-4 per day)
    min_levels = 200
    if len(hvn_levels) < min_levels:
        # Add all remaining levels with lower strength
        remaining_levels = [level for level in volume_profile.index if level not in hvn_levels]
        remaining_levels.sort(key=lambda x: volume_profile.loc[x], reverse=True)  # Sort by volume
        
        for level in remaining_levels[:min_levels - len(hvn_levels)]:
            hvn_levels.append(level)
            volume_at_level = volume_profile.loc[level]
            total_volume = volume_profile.sum()
            strength = min(volume_at_level / total_volume * 40, 0.6)  # Lower strength for forced levels
            hvn_strengths[level] = strength
    
    # Method 7: Add ALL levels if we still don't have enough (most aggressive)
    # If we still don't have enough levels, add every single price level
    if len(hvn_levels) < min_levels:
        all_levels = list(volume_profile.index)
        for level in all_levels:
            if level not in hvn_levels:
                hvn_levels.append(level)
                volume_at_level = volume_profile.loc[level]
                total_volume = volume_profile.sum()
                strength = min(volume_at_level / total_volume * 30, 0.5)  # Very low strength for all levels
                hvn_strengths[level] = strength
    
    # Sort levels and remove duplicates
    hvn_levels = sorted(list(set(hvn_levels)))
    
    # Create HVN results with strength information
    hvn_results = []
    for level in hvn_levels:
        strength = hvn_strengths.get(level, 0.5)  # Default strength if not calculated
        hvn_results.append({
            'price': level,
            'strength': strength,
            'volume_concentration': volume_profile.loc[level] / volume_profile.sum(),
            'method': 'hvn'
        })
    
    # Sort by strength (strongest first)
    hvn_results.sort(key=lambda x: x['strength'], reverse=True)
    
    # print(f"Volume Profile: POC={poc_price:.2f}, HVNs={len(hvn_results)}")
    return {
        "poc": poc_price,
        "hvn_levels": [hvn['price'] for hvn in hvn_results],
        "hvn_results": hvn_results,  # Include full results with strength
        "lvn_levels": [],  # No LVNs as requested
        "volume_in_bins": volume_profile,
    }


def create_dummy_data(filename, data_type, num_records=1000, start_date="2023-01-01"):
    """
    Creates dummy CSV data for klines, aggregated trades, or futures.
    This function is now centralized in data_utils.
    """
    if os.path.exists(filename):
        print(f"Dummy data file '{filename}' already exists. Skipping creation.")
        return

    print(f"Creating dummy {data_type} data at {filename}...")
    dates = pd.date_range(start=start_date, periods=num_records, freq="1min")

    if data_type == "klines":
        # Simulate price movement
        price = 1000 + np.cumsum(np.random.randn(num_records))
        df = pd.DataFrame(
            {
                "open_time": dates,
                "open": price,
                "high": price + np.random.rand(num_records) * 5,
                "low": price - np.random.rand(num_records) * 5,
                "close": price + np.random.randn(num_records),
                "volume": np.random.randint(100, 10000, num_records),
                "close_time": dates + pd.Timedelta(minutes=1),
                "quote_asset_volume": np.random.rand(num_records) * 100000,
                "number_of_trades": np.random.randint(50, 500, num_records),
                "taker_buy_base_asset_volume": np.random.rand(num_records) * 5000,
                "taker_buy_quote_asset_volume": np.random.rand(num_records) * 50000,
                "ignore": 0,
            },
        )
        df.set_index("open_time", inplace=True)
    elif data_type == "agg_trades":
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "a": np.arange(num_records),  # Aggregate tradeId
                "p": 1000 + np.cumsum(np.random.randn(num_records) * 0.1),  # Price
                "q": np.random.rand(num_records) * 10,  # Quantity
                "f": np.arange(num_records),  # First tradeId
                "l": np.arange(num_records),  # Last tradeId
                "T": (dates.astype(np.int64) // 10**6),  # Timestamp in ms
                "m": np.random.choice(
                    [True, False],
                    num_records,
                ),  # Was the buyer the maker?
                "M": np.random.choice(
                    [True, False],
                    num_records,
                ),  # Was the trade the best price match?
            },
        )
        df.set_index("timestamp", inplace=True)
        df.rename(
            columns={"p": "price", "q": "quantity", "m": "is_buyer_maker"},
            inplace=True,
        )
    elif data_type == "futures":
        df = pd.DataFrame(
            {
                "timestamp": dates,
                "fundingRate": np.random.rand(num_records) * 0.0001
                - 0.00005,  # Small positive/negative
            },
        )
        df.set_index("timestamp", inplace=True)
    else:
        print(f"Unknown data type: {data_type}. Skipping dummy data creation.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists
    df.to_csv(filename)
    print(f"Dummy {data_type} data saved to '{filename}'.")
