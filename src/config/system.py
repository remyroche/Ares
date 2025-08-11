# src/config/system.py

import os
from typing import Any

from src.config.environment import get_environment_settings


def get_system_config() -> dict[str, Any]:
    """
    Get the complete system configuration.

    Returns:
        dict: Complete system configuration
    """
    settings = get_environment_settings()

    return {
        # --- Logging Configuration ---
        "logging": {
            "level": settings.log_level,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "console_output": True,
            "file_output": True,
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "backup_count": 5,
            "log_directory": "log",
            "enable_rotation": True,
            "enable_timestamped_files": True,
            "enable_global_logging": True,  # Global log file per session
            "enable_error_logging": True,
            "enable_performance_logging": True,
            "enable_trade_logging": True,
            "enable_system_logging": True,
        },
        # --- Database Configuration ---
        "database": {
            "type": "sqlite",  # 'sqlite' only - Firebase removed
            "sqlite_db_path": "data/ares_local_db.sqlite",  # Path for SQLite database file
            "backup_interval_hours": 24,  # Automatic backup interval
            "backup_retention_days": 30,  # How long to keep backups
            "db_type": os.getenv("DB_TYPE", "influxdb"),  # Moved inside CONFIG
            # InfluxDB Configuration
            "influxdb": {
                "url": settings.influxdb_url,
                "token": settings.influxdb_token,
                "org": settings.influxdb_org,
                "bucket": settings.influxdb_bucket,
            },
            # Firestore Configuration
            "firestore": {
                "credentials": settings.google_application_credentials,
                "project_id": settings.firestore_project_id,
            },
        },
        # --- File Paths and Data Configuration ---
        "data": {
            # Data Caching Configuration (filenames are set dynamically below)
            "klines_filename": "",
            "agg_trades_filename": "",
            "futures_filename": "",
            "prepared_data_filename": "",
            # Script Names
            "downloader_script_name": "backtesting/ares_data_downloader.py",
            "preparer_script_name": "backtesting/ares_data_preparer.py",
            "pipeline_script_name": "src/ares_pipeline.py",
            "pipeline_pid_file": "ares_pipeline.pid",
            "restart_flag_file": "restart_pipeline.flag",
        },
        # --- Checkpointing Configuration ---
        "checkpointing": {
            "checkpoint_dir": "checkpoints",
            "optimizer_checkpoint_file": "optimizer_state.pkl",  # Relative to CHECKPOINT_DIR
            "pipeline_progress_file": "pipeline_progress.json",  # Relative to CHECKPOINT_DIR
            "walk_forward_reports_file": "walk_forward_reports.json",  # Relative to CHECKPOINT_DIR
            "prepared_data_checkpoint_file": "full_prepared_data.parquet",  # Relative to CHECKPOINT_DIR
            "regime_classifier_model_prefix": "regime_classifier_fold_",  # Prefix for fold-specific models
            "ensemble_model_prefix": "ensemble_fold_",  # Prefix for fold-specific ensemble models
        },
        # --- Reporting Configuration ---
        "reporting": {
            "detailed_trade_log_file": "reports/detailed_trade_log.csv",
            "daily_summary_log_filename_format": "reports/daily_summary_log_%Y-%m.csv",  # Monthly filenames for summary reports
            "strategy_performance_log_filename_format": "reports/strategy_performance_log_%Y-%m.csv",  # Monthly filenames for strategy reports
            "error_log_file": "ares_errors.jsonl",  # Dedicated error log file
        },
        # --- MLflow Configuration ---
        "mlflow": {
            "tracking_uri": settings.mlflow_tracking_uri,
            "experiment_name": settings.mlflow_experiment_name,
        },
        # --- Version Information ---
        "version": {
            "ares_version": "2.0.0",
            "log_level": settings.log_level,
        },
    }


def get_logging_config() -> dict[str, Any]:
    """
    Get logging configuration.

    Returns:
        dict: Logging configuration
    """
    system_config = get_system_config()
    return system_config.get("logging", {})


def get_database_config() -> dict[str, Any]:
    """
    Get database configuration.

    Returns:
        dict: Database configuration
    """
    system_config = get_system_config()
    return system_config.get("database", {})


def get_data_config() -> dict[str, Any]:
    """
    Get data configuration.

    Returns:
        dict: Data configuration
    """
    system_config = get_system_config()
    return system_config.get("data", {})


def get_checkpointing_config() -> dict[str, Any]:
    """
    Get checkpointing configuration.

    Returns:
        dict: Checkpointing configuration
    """
    system_config = get_system_config()
    return system_config.get("checkpointing", {})


def get_reporting_config() -> dict[str, Any]:
    """
    Get reporting configuration.

    Returns:
        dict: Reporting configuration
    """
    system_config = get_system_config()
    return system_config.get("reporting", {})


def get_mlflow_config() -> dict[str, Any]:
    """
    Get MLflow configuration.

    Returns:
        dict: MLflow configuration
    """
    system_config = get_system_config()
    return system_config.get("mlflow", {})


def get_version_info() -> dict[str, Any]:
    """
    Get version information.

    Returns:
        dict: Version information
    """
    system_config = get_system_config()
    return system_config.get("version", {})


def get_log_level() -> str:
    """
    Get the current log level.

    Returns:
        str: The log level
    """
    logging_config = get_logging_config()
    return logging_config.get("level", "INFO")


def get_log_directory() -> str:
    """
    Get the log directory.

    Returns:
        str: The log directory
    """
    logging_config = get_logging_config()
    return logging_config.get("log_directory", "log")


def get_database_type() -> str:
    """
    Get the database type.

    Returns:
        str: The database type
    """
    database_config = get_database_config()
    return database_config.get("type", "sqlite")


def get_sqlite_db_path() -> str:
    """
    Get the SQLite database path.

    Returns:
        str: The SQLite database path
    """
    database_config = get_database_config()
    return database_config.get("sqlite_db_path", "data/ares_local_db.sqlite")


def get_checkpoint_dir() -> str:
    """
    Get the checkpoint directory.

    Returns:
        str: The checkpoint directory
    """
    checkpointing_config = get_checkpointing_config()
    return checkpointing_config.get("checkpoint_dir", "checkpoints")


def get_ares_version() -> str:
    """
    Get the Ares version.

    Returns:
        str: The Ares version
    """
    version_info = get_version_info()
    return version_info.get("ares_version", "2.0.0")
