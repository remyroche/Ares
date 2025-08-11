"""
Training validation configuration and rules.
Defines error thresholds, validation criteria, and step progression rules.
"""

import os
from typing import Any

import numpy as np
import pandas as pd

# Critical error thresholds for each training step
CRITICAL_ERROR_THRESHOLDS = {
    "setup": {
        "max_setup_time": 60,  # seconds
        "required_components": ["database", "efficiency_optimizer", "data_directory"],
        "min_disk_space_mb": 1000,  # MB
        "min_memory_mb": 512,  # MB
    },
    "data_collection": {
        "min_data_rows": 1000,
        "max_missing_percentage": 1.0,  # Changed from 50.0 to 1.0
        "required_columns": ["open", "high", "low", "close", "volume"],
        "min_data_quality_score": 0.7,
        "max_data_collection_time": 300,  # 5 minutes
    },
    "preliminary_optimization": {
        "min_trials_completed": 10,  # Increased from 1 to 10
        "max_optimization_time": 1800,  # Increased to 30 minutes (1800 seconds)
        "min_features_available": 5,
        "min_optimization_score": -1.0,  # Allow negative scores but not too bad
        "required_output_files": ["optimal_target_params.json"],
    },
    "coarse_optimization": {
        "min_features_pruned": 3,
        "max_optimization_time": 3600,  # Increased to 60 minutes for 2 years of data
        "min_sharpe_ratio": 0.5,  # Increased from 0.1 to 0.5
        "min_profit_factor": 1.3,  # Increased from 1.1 to 1.3
        "min_features_remaining": 2,
        "required_output_files": ["pruned_features.json", "hpo_ranges.json"],
    },
    "main_model_training": {
        "min_sharpe_ratio": 0.8,  # Increased from 0.2 to 0.8
        "max_training_time": 86400,  # Increased to 24 hours (86400 seconds)
        "min_profit_factor": 1.5,  # Increased from 1.2 to 1.5
        "max_overfitting_threshold": 0.2,
        "required_output_files": ["model.pkl", "scaler.pkl"],
    },
    "multi_stage_hpo": {
        "min_stages_completed": 2,
        "max_total_trials": 200,
        "min_best_score": 0.6,
        "max_hpo_time": 3600,  # 1 hour
        "required_output_files": ["best_hyperparameters.json"],
    },
    "walk_forward_validation": {
        "min_windows": 3,
        "min_validation_score": 0.5,
        "max_validation_time": 1200,  # 20 minutes
        "required_output_files": ["walk_forward_results.json"],
    },
    "monte_carlo_validation": {
        "min_simulations": 100,
        "min_confidence_interval": 0.8,
        "max_validation_time": 900,  # 15 minutes
        "required_output_files": ["monte_carlo_results.json"],
    },
    "ab_testing_setup": {
        "min_test_groups": 2,
        "max_setup_time": 300,  # 5 minutes
        "required_output_files": ["ab_test_config.json"],
    },
    "save_results": {
        "max_save_time": 60,  # 1 minute
        "required_output_files": ["training_summary.json", "model_artifacts.zip"],
    },
}

# Step progression rules
STEP_PROGRESSION_RULES = {
    "setup": {
        "can_skip": False,
        "required_for": [],  # No steps required for setup
        "failure_action": "STOP_PIPELINE",
    },
    "data_collection": {
        "can_skip": False,
        "required_for": ["setup"],  # Setup is required for data_collection
        "failure_action": "STOP_PIPELINE",
    },
    "preliminary_optimization": {
        "can_skip": True,
        "required_for": [
            "data_collection",
        ],  # Data collection is required for preliminary_optimization
        "failure_action": "SKIP_DEPENDENT_STEPS",
    },
    "coarse_optimization": {
        "can_skip": True,
        "required_for": [
            "preliminary_optimization",
        ],  # Preliminary optimization is required for coarse_optimization
        "failure_action": "SKIP_DEPENDENT_STEPS",
    },
    "main_model_training": {
        "can_skip": True,
        "required_for": [
            "coarse_optimization",
        ],  # Coarse optimization is required for main_model_training
        "failure_action": "SKIP_DEPENDENT_STEPS",
    },
    "multi_stage_hpo": {
        "can_skip": True,
        "required_for": [
            "main_model_training",
        ],  # Main model training is required for multi_stage_hpo
        "failure_action": "CONTINUE_WITH_WARNING",
    },
    "walk_forward_validation": {
        "can_skip": True,
        "required_for": [
            "multi_stage_hpo",
        ],  # Multi-stage HPO is required for walk_forward_validation
        "failure_action": "CONTINUE_WITH_WARNING",
    },
    "monte_carlo_validation": {
        "can_skip": True,
        "required_for": [
            "walk_forward_validation",
        ],  # Walk forward validation is required for monte_carlo_validation
        "failure_action": "CONTINUE_WITH_WARNING",
    },
    "ab_testing_setup": {
        "can_skip": True,
        "required_for": [
            "monte_carlo_validation",
        ],  # Monte Carlo validation is required for ab_testing_setup
        "failure_action": "CONTINUE_WITH_WARNING",
    },
    "save_results": {
        "can_skip": False,
        "required_for": [
            "ab_testing_setup",
        ],  # AB testing setup is required for save_results
        "failure_action": "STOP_PIPELINE",
    },
}

# Error severity levels and their impact
ERROR_SEVERITY_LEVELS = {
    "CRITICAL": {
        "description": "Step cannot proceed, pipeline should stop",
        "action": "STOP_STEP",
        "can_skip": False,
    },
    "ERROR": {
        "description": "Step failed but can be skipped",
        "action": "SKIP_STEP",
        "can_skip": True,
    },
    "WARNING": {
        "description": "Step completed with issues",
        "action": "CONTINUE_WITH_WARNING",
        "can_skip": False,
    },
    "INFO": {
        "description": "Informational message",
        "action": "CONTINUE",
        "can_skip": False,
    },
}


class DataValidator:
    """Class to handle data validation with focused methods."""

    def __init__(self):
        self.errors = []

    def validate_data_format(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate data format and structure."""
        self.errors = []

        # Check if data is a dictionary
        if not isinstance(data, dict):
            self.errors.append("Data must be a dictionary")
            return False, self.errors

        # Check for required keys
        required_keys = ["klines", "agg_trades", "futures"]
        for key in required_keys:
            if key not in data:
                self.errors.append(f"Missing required key: {key}")

        # Validate each data type
        self._validate_klines_format(data.get("klines"))
        self._validate_agg_trades_format(data.get("agg_trades"))
        self._validate_futures_format(data.get("futures"))

        return len(self.errors) == 0, self.errors

    def _validate_klines_format(self, klines) -> None:
        """Validate klines data format."""
        if klines is None:
            return

        if not isinstance(klines, pd.DataFrame):
            self.errors.append("klines must be a pandas DataFrame")
            return

        # Check for required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in klines.columns]
        if missing_columns:
            self.errors.append(f"Missing required columns in klines: {missing_columns}")

        # Check for proper data types
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in klines.columns:
                if not pd.api.types.is_numeric_dtype(klines[col]):
                    self.errors.append(f"Column {col} must be numeric")

        # Check for datetime index
        if not isinstance(klines.index, pd.DatetimeIndex):
            self.errors.append("klines must have a DatetimeIndex")

    def _validate_agg_trades_format(self, agg_trades) -> None:
        """Validate aggregated trades data format."""
        if agg_trades is None:
            return

        if not isinstance(agg_trades, pd.DataFrame):
            self.errors.append("agg_trades must be a pandas DataFrame")
            return

        # Check for required columns
        required_columns = ["price", "quantity", "is_buyer_maker"]
        missing_columns = [
            col for col in required_columns if col not in agg_trades.columns
        ]
        if missing_columns:
            self.errors.append(
                f"Missing required columns in agg_trades: {missing_columns}",
            )

    def _validate_futures_format(self, futures) -> None:
        """Validate futures data format."""
        if futures is None:
            return

        if not isinstance(futures, pd.DataFrame):
            self.errors.append("futures must be a pandas DataFrame")
            return

        if "fundingRate" not in futures.columns:
            self.errors.append("futures must have 'fundingRate' column")

    def validate_data_quality(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate data quality and integrity."""
        self.errors = []

        if "klines" in data and isinstance(data["klines"], pd.DataFrame):
            self._validate_klines_quality(data["klines"])

        if "agg_trades" in data and isinstance(data["agg_trades"], pd.DataFrame):
            self._validate_agg_trades_quality(data["agg_trades"])

        if "futures" in data and isinstance(data["futures"], pd.DataFrame):
            self._validate_futures_quality(data["futures"])

        return len(self.errors) == 0, self.errors

    def _validate_klines_quality(self, klines: pd.DataFrame) -> None:
        """Validate klines data quality."""
        # Check for infinite values
        for col in ["open", "high", "low", "close", "volume"]:
            if col in klines.columns:
                if klines[col].isin([np.inf, -np.inf]).any():
                    self.errors.append(f"Column {col} contains infinite values")

        # Check for negative prices
        price_columns = ["open", "high", "low", "close"]
        for col in price_columns:
            if col in klines.columns and (klines[col] <= 0).any():
                self.errors.append(f"Column {col} contains non-positive values")

        # Check for negative volume
        if "volume" in klines.columns and (klines["volume"] < 0).any():
            self.errors.append("Volume contains negative values")

        # Check for logical inconsistencies
        if all(col in klines.columns for col in ["high", "low"]):
            if (klines["high"] < klines["low"]).any():
                self.errors.append("High values are less than low values")

    def _validate_agg_trades_quality(self, agg_trades: pd.DataFrame) -> None:
        """Validate aggregated trades data quality."""
        # Check for negative prices
        if "price" in agg_trades.columns and (agg_trades["price"] <= 0).any():
            self.errors.append("Aggregated trades contain non-positive prices")

        # Check for negative quantities
        if "quantity" in agg_trades.columns:
            if (agg_trades["quantity"] < 0).any():
                self.errors.append("Aggregated trades contain negative quantities")

    def _validate_futures_quality(self, futures: pd.DataFrame) -> None:
        """Validate futures data quality."""
        # Check for infinite funding rates
        if "fundingRate" in futures.columns:
            if futures["fundingRate"].isin([np.inf, -np.inf]).any():
                self.errors.append("Futures contain infinite funding rates")


# Create global validator instance
_data_validator = DataValidator()


def validate_data_format(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate data format and structure."""
    return _data_validator.validate_data_format(data)


def validate_data_quality(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate data quality and integrity."""
    return _data_validator.validate_data_quality(data)


def validate_imports() -> tuple[bool, list[str]]:
    """Validate that all required imports are available."""
    errors = []

    # Required modules (critical for operation)
    required_modules = [
        "pandas",
        "numpy",
        "sklearn",
        "optuna",
        "lightgbm",
        "xgboost",
        "catboost",
        "matplotlib",
        "seaborn",  # seaborn is required for data quality analysis
    ]

    # Check required modules
    for module in required_modules:
        try:
            __import__(module)
        except ImportError as e:
            errors.append(f"Missing required module: {module} - {str(e)}")

    return len(errors) == 0, errors


def validate_file_paths(data_dir: str) -> tuple[bool, list[str]]:
    """Validate that required file paths exist and are accessible."""
    errors = []

    # Check if data directory exists
    if not os.path.exists(data_dir):
        errors.append(f"Data directory does not exist: {data_dir}")
        return False, errors

    # Check if data directory is writable
    if not os.access(data_dir, os.W_OK):
        errors.append(f"Data directory is not writable: {data_dir}")

    # Check for required subdirectories
    required_dirs = ["cache", "models", "logs"]
    for subdir in required_dirs:
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(subdir_path):
            try:
                os.makedirs(subdir_path, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create required directory {subdir}: {str(e)}")

    return len(errors) == 0, errors


def validate_system_resources() -> tuple[bool, list[str]]:
    """Validate system resources are sufficient."""
    errors = []

    import psutil

    # Check available memory (need at least 2GB free for blank mode, 4GB for full training)
    memory = psutil.virtual_memory()

    # Check if we're in blank training mode by looking at environment or config
    import os

    blank_mode = os.getenv("BLANK_TRAINING_MODE", "0") == "1"

    # Debug logging
    print(
        f"🔍 DEBUG: BLANK_TRAINING_MODE environment variable: {os.getenv('BLANK_TRAINING_MODE', 'not set')}",
    )
    print(f"🔍 DEBUG: blank_mode detected: {blank_mode}")
    print(f"🔍 DEBUG: Available memory: {memory.available / (1024**3):.1f}GB")

    if blank_mode:
        # More lenient requirements for blank mode
        min_memory_gb = 2
        min_disk_gb = 5
        min_cpu_cores = 2
        print(
            f"🔍 DEBUG: Using blank mode requirements: {min_memory_gb}GB RAM, {min_disk_gb}GB disk, {min_cpu_cores} CPU cores",
        )
    else:
        # Full requirements for production training
        min_memory_gb = 4
        min_disk_gb = 10
        min_cpu_cores = 4
        print(
            f"🔍 DEBUG: Using production requirements: {min_memory_gb}GB RAM, {min_disk_gb}GB disk, {min_cpu_cores} CPU cores",
        )

    if memory.available < min_memory_gb * 1024 * 1024 * 1024:
        errors.append(
            f"Insufficient memory: {memory.available / (1024**3):.1f}GB available, need {min_memory_gb}GB",
        )

    # Check available disk space
    disk = psutil.disk_usage("/")
    if disk.free < min_disk_gb * 1024 * 1024 * 1024:
        errors.append(
            f"Insufficient disk space: {disk.free / (1024**3):.1f}GB available, need {min_disk_gb}GB",
        )

    # Check CPU cores
    cpu_count = psutil.cpu_count()
    if cpu_count < min_cpu_cores:
        errors.append(
            f"Insufficient CPU cores: {cpu_count} available, need {min_cpu_cores}",
        )

    return len(errors) == 0, errors


def validate_data_collection(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate data collection step results."""
    errors = []

    # Check if data exists
    if not data:
        errors.append("No data collected")
        return False, errors

    # Check if required data types are present
    required_keys = ["klines", "agg_trades", "futures"]
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required data type: {key}")
        elif data[key] is None or (hasattr(data[key], "empty") and data[key].empty):
            errors.append(f"Empty data for: {key}")

    # Check data quality
    if "klines" in data and data["klines"] is not None:
        klines = data["klines"]
        if hasattr(klines, "shape") and klines.shape[0] < 1000:
            errors.append("Insufficient klines data (need at least 1000 rows)")

    return len(errors) == 0, errors


def validate_preliminary_optimization(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate preliminary optimization step results."""
    errors = []

    # Check if optimization results exist
    if not data:
        errors.append("No optimization results")
        return False, errors

    # Check if required parameters are present
    required_params = ["tp_threshold", "sl_threshold", "holding_period"]
    for param in required_params:
        if param not in data:
            errors.append(f"Missing required parameter: {param}")

    return len(errors) == 0, errors


def validate_coarse_optimization(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate coarse optimization step results."""
    errors = []

    # Check if optimization results exist
    if not data:
        errors.append("No coarse optimization results")
        return False, errors

    # Check if we have reasonable number of parameters
    # For blank training: at least 3 parameters
    # For production: at least 8 parameters (8-12 optimal range)
    min_params = 3  # Conservative minimum for any mode
    production_min_params = 8  # Production mode minimum

    # Check if we're in production mode (more than 5 parameters suggests production)
    is_production_mode = len(data) >= 5

    if is_production_mode and len(data) < production_min_params:
        found_params = list(data.keys())
        errors.append(
            f"Production mode requires at least {production_min_params} parameters. Found: {found_params}",
        )

    elif len(data) < min_params:
        found_params = list(data.keys())
        errors.append(
            f"Too few parameters found. Found: {found_params} (need at least {min_params})",
        )

    # Validate parameter structure
    for param_name, param_config in data.items():
        if not isinstance(param_config, dict):
            errors.append(f"Invalid parameter config for {param_name}")
            continue

        # Check for required keys in parameter config
        required_keys = ["low", "high", "type"]
        missing_keys = [key for key in required_keys if key not in param_config]
        if missing_keys:
            errors.append(f"Missing keys for {param_name}: {missing_keys}")

    return len(errors) == 0, errors


# Enhanced validation function mapping
VALIDATION_FUNCTIONS = {
    "data_collection": validate_data_collection,
    "preliminary_optimization": validate_preliminary_optimization,
    "coarse_optimization": validate_coarse_optimization,
    "imports": validate_imports,
    "data_format": validate_data_format,
    "data_quality": validate_data_quality,
    "file_paths": validate_file_paths,
    "system_resources": validate_system_resources,
}


def get_validation_config(step_name: str) -> dict[str, Any]:
    """Get validation configuration for a specific step."""
    return CRITICAL_ERROR_THRESHOLDS.get(step_name, {})


def get_progression_rules(step_name: str) -> dict[str, Any]:
    """Get progression rules for a specific step."""
    return STEP_PROGRESSION_RULES.get(step_name, {})


def can_proceed_to_step(
    current_step: str,
    next_step: str,
    step_status: dict[str, Any],
) -> tuple[bool, str]:
    """Check if we can proceed to the next step based on current step status."""
    current_rules = get_progression_rules(current_step)
    next_rules = get_progression_rules(next_step)

    # Check if current step is required for next step
    if current_step in next_rules.get("required_for", []):
        if step_status.get(current_step, {}).get("status") == "FAILED" and (
            current_rules.get("failure_action") == "STOP_PIPELINE"
            or current_rules.get("failure_action") == "SKIP_DEPENDENT_STEPS"
        ):
            return (
                False,
                f"Cannot proceed to {next_step}: {current_step} failed and is required",
            )

    return True, f"Proceeding to {next_step}"
