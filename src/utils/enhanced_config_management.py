from __future__ import annotations

"""
Enhanced Configuration Management Utilities

This module provides structured configuration management for the training pipeline.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from src.utils.logger import system_logger
    from src.utils.pipeline_standards import PipelineStandards as PipelineStandards_src_utils_pipeline_standards
    from src.utils.pipeline_standards import pipeline_standards
except ImportError:
    system_logger = logging.getLogger("EnhancedConfigManagement")


@dataclass
class Step1Config:
    """Enhanced configuration for Step1 data collection."""

    # Basic parameters
    symbol: str = "ETHUSDT"
    exchange: str = "BINANCE"
    timeframe: str = "1m"
    lookback_days: int = 1095

    # Performance parameters
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    chunk_size: int = 10000
    max_memory_mb: int = 1024
    max_workers: int = 4

    # Quality thresholds
    max_nan_ratio: float = 0.0  # Zero tolerance for NaN
    max_infinite_count: int = 0  # Zero tolerance for infinite values
    min_unique_values: int = 2
    max_gap_hours: int = 48
    price_tolerance: float = 0.001
    volume_tolerance: float = 0.001

    # Data directories
    data_dir: str = "data_cache"
    backup_dir: str = "data_cache/backup"
    temp_dir: str = "data_cache/temp"

    # Error handling
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0

    def validate(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []

        if self.lookback_days <= 0:
            issues.append("lookback_days must be positive")
        if self.chunk_size <= 0:
            issues.append("chunk_size must be positive")
        if self.max_memory_mb <= 0:
            issues.append("max_memory_mb must be positive")
        if self.max_retries < 0:
            issues.append("max_retries must be non-negative")
        if self.max_nan_ratio < 0 or self.max_nan_ratio > 1:
            issues.append("max_nan_ratio must be between 0 and 1")
        if self.max_infinite_count < 0:
            issues.append("max_infinite_count must be non-negative")
        if self.price_tolerance < 0:
            issues.append("price_tolerance must be non-negative")
        if self.volume_tolerance < 0:
            issues.append("volume_tolerance must be non-negative")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Step1Config":
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class Step1_5Config:
    """Enhanced configuration for Step1_5 data converter."""

    # Basic parameters
    symbol: str = "ETHUSDT"
    exchange: str = "BINANCE"
    timeframe: str = "1m"

    # Performance parameters
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    chunk_size: int = 10000
    max_memory_mb: int = 1024
    max_workers: int = 4
    batch_size: int = 262144

    # Quality thresholds
    max_nan_ratio: float = 0.0  # Zero tolerance for NaN
    max_infinite_count: int = 0  # Zero tolerance for infinite values
    min_unique_values: int = 2
    max_gap_hours: int = 48
    price_tolerance: float = 0.001
    volume_tolerance: float = 0.001

    # Data directories
    data_dir: str = "data_cache"
    unified_dir: str = "data_cache/unified"
    backup_dir: str = "data_cache/backup_pre_unified"
    temp_dir: str = "data_cache/temp"

    # Processing options
    force_rerun: bool = False
    enable_incremental: bool = True
    auto_add_date_columns: bool = True
    compression: str = "snappy"
    use_dictionary: bool = True
    min_rows_per_group: int = 50000
    max_rows_per_file: int = 5_000_000

    # Error handling
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0

    def validate(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []

        if self.chunk_size <= 0:
            issues.append("chunk_size must be positive")
        if self.max_memory_mb <= 0:
            issues.append("max_memory_mb must be positive")
        if self.max_retries < 0:
            issues.append("max_retries must be non-negative")
        if self.max_nan_ratio < 0 or self.max_nan_ratio > 1:
            issues.append("max_nan_ratio must be between 0 and 1")
        if self.min_rows_per_group >= self.max_rows_per_file:
            issues.append("min_rows_per_group must be less than max_rows_per_file")
        if self.price_tolerance < 0:
            issues.append("price_tolerance must be non-negative")
        if self.volume_tolerance < 0:
            issues.append("volume_tolerance must be non-negative")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Step1_5Config":
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline."""

    # Step configurations
    step01: Step1Config = field(default_factory=Step1Config)
    step1_5: Step1_5Config = field(default_factory=Step1_5Config)

    # Global settings
    environment: str = "development"
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_profiling: bool = False

    # Data settings
    default_symbol: str = "ETHUSDT"
    default_exchange: str = "BINANCE"
    default_timeframe: str = "1m"

    def validate(self) -> List[str]:
        """Validate pipeline configuration."""
        issues = []

        # Validate individual step configurations
        step1_issues = self.step01.validate()
        issues.extend([f"step01.{issue}" for issue in step1_issues])

        step1_5_issues = self.step1_5.validate()
        issues.extend([f"step1_5.{issue}" for issue in step1_5_issues])

        # Validate global settings
        if self.environment not in ["development", "staging", "production"]:
            issues.append(
                "environment must be one of: development, staging, production"
            )

        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            issues.append("log_level must be a valid logging level")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "step01": self.step01.to_dict(),
            "step1_5": self.step1_5.to_dict(),
            "environment": self.environment,
            "log_level": self.log_level,
            "enable_metrics": self.enable_metrics,
            "enable_profiling": self.enable_profiling,
            "default_symbol": self.default_symbol,
            "default_exchange": self.default_exchange,
            "default_timeframe": self.default_timeframe,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary."""
        step1_config = Step1Config.from_dict(config_dict.get("step01", {}))
        step1_5_config = Step1_5Config.from_dict(config_dict.get("step1_5", {}))

        return cls(
            step01=step1_config,
            step1_5=step1_5_config,
            environment=config_dict.get("environment", "development"),
            log_level=config_dict.get("log_level", "INFO"),
            enable_metrics=config_dict.get("enable_metrics", True),
            enable_profiling=config_dict.get("enable_profiling", False),
            default_symbol=config_dict.get("default_symbol", "ETHUSDT"),
            default_exchange=config_dict.get("default_exchange", "BINANCE"),
            default_timeframe=config_dict.get("default_timeframe", "1m"),
        )


class ConfigManager:
    """Manager for configuration loading, validation, and saving."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = system_logger.getChild("ConfigManager")

    def load_config(self, config_name: str = "pipeline_config.json") -> PipelineConfig:
        """Load configuration from file."""
        config_path = self.config_dir / config_name

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config_dict = json.load(f)

                config = PipelineConfig.from_dict(config_dict)
                self.logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                self.logger.warning(
                    f"Error loading configuration from {config_path}: {e}"
                )
                self.logger.info("Using default configuration")

        # Return default configuration
        return PipelineConfig()

    def save_config(
        self, config: PipelineConfig, config_name: str = "pipeline_config.json"
    ):
        """Save configuration to file."""
        config_path = self.config_dir / config_name

        try:
            config_dict = config.to_dict()
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

            self.logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration to {config_path}: {e}")

    def validate_config(self, config: PipelineConfig) -> bool:
        """Validate configuration and log any issues."""
        issues = config.validate()

        if issues:
            self.logger.error("Configuration validation failed:")
            for issue in issues:
                self.logger.error(f"  - {issue}")
            return False
        else:
            self.logger.info("Configuration validation passed")
            return True

    def create_environment_config(self, environment: str) -> PipelineConfig:
        """Create environment-specific configuration."""
        base_config = PipelineConfig()

        if environment == "development":
            base_config.environment = "development"
            base_config.log_level = "DEBUG"
            base_config.enable_profiling = True
            base_config.step01.max_memory_mb = 512
            base_config.step1_5.max_memory_mb = 512
            base_config.step01.chunk_size = 5000
            base_config.step1_5.chunk_size = 5000

        elif environment == "staging":
            base_config.environment = "staging"
            base_config.log_level = "INFO"
            base_config.enable_profiling = False
            base_config.step01.max_memory_mb = 2048
            base_config.step1_5.max_memory_mb = 2048

        elif environment == "production":
            base_config.environment = "production"
            base_config.log_level = "WARNING"
            base_config.enable_profiling = False
            base_config.step01.max_memory_mb = 4096
            base_config.step1_5.max_memory_mb = 4096
            base_config.step01.max_retries = 5
            base_config.step1_5.max_retries = 5

        return base_config

    def load_environment_config(self, environment: str) -> PipelineConfig:
        """Load environment-specific configuration."""
        config_name = f"pipeline_config_{environment}.json"
        config = self.load_config(config_name)

        if config.environment != environment:
            # Create new environment-specific config
            config = self.create_environment_config(environment)
            self.save_config(config, config_name)

        return config


# Convenience functions
def get_default_step1_config() -> Step1Config:
    """Get default Step1 configuration."""
    return Step1Config()


def get_default_step1_5_config() -> Step1_5Config:
    """Get default Step1_5 configuration."""
    return Step1_5Config()


def get_default_pipeline_config() -> PipelineConfig:
    """Get default pipeline configuration."""
    return PipelineConfig()


def load_pipeline_config(environment: str = "development") -> PipelineConfig:
    """Load pipeline configuration for specified environment."""
    config_manager = ConfigManager()
    return config_manager.load_environment_config(environment)


def validate_and_save_config(
    config: PipelineConfig, config_name: str = "pipeline_config.json"
):
    """Validate and save configuration."""
    config_manager = ConfigManager()

    if config_manager.validate_config(config):
        config_manager.save_config(config, config_name)
        return True
    else:
        return False


# Environment-specific configuration presets
DEVELOPMENT_CONFIG = {
    "step01": {
        "max_memory_mb": 512,
        "chunk_size": 5000,
        "max_retries": 2,
        "log_level": "DEBUG",
    },
    "step1_5": {
        "max_memory_mb": 512,
        "chunk_size": 5000,
        "max_retries": 2,
        "enable_incremental": True,
    },
    "environment": "development",
    "log_level": "DEBUG",
    "enable_profiling": True,
}

STAGING_CONFIG = {
    "step01": {"max_memory_mb": 2048, "chunk_size": 10000, "max_retries": 3},
    "step1_5": {
        "max_memory_mb": 2048,
        "chunk_size": 10000,
        "max_retries": 3,
        "enable_incremental": True,
    },
    "environment": "staging",
    "log_level": "INFO",
    "enable_profiling": False,
}

PRODUCTION_CONFIG = {
    "step01": {
        "max_memory_mb": 4096,
        "chunk_size": 20000,
        "max_retries": 5,
        "circuit_breaker_failure_threshold": 3,
    },
    "step1_5": {
        "max_memory_mb": 4096,
        "chunk_size": 20000,
        "max_retries": 5,
        "enable_incremental": True,
        "circuit_breaker_failure_threshold": 3,
    },
    "environment": "production",
    "log_level": "WARNING",
    "enable_profiling": False,
}
