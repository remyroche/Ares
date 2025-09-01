# src/training/enhanced_lm_config.py

"""Pydantic-based configuration for Enhanced LM Optimizer.

This module provides type-safe configuration with automatic validation,
clear error messages, and auto-generated documentation.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class SamplerType(str, Enum):
    """Available Optuna samplers."""

    TPE = "tpe"
    CMAES = "cmaes"
    RANDOM = "random"


class PrunerType(str, Enum):
    """Available Optuna pruners."""

    MEDIAN = "median"
    HYPERBAND = "hyperband"
    THRESHOLD = "threshold"


class FeatureSelectionConfig(BaseModel):
    """Configuration for feature selection."""

    enable: bool = Field(default=True, description="Enable feature selection")
    methods: list[str] = Field(
        default=["mutual_info", "lasso", "random_forest", "shap"],
        description="Feature selection methods to use",
    )
    target_features: dict[str, int] = Field(
        default={"step6": 80, "step06_5": 100, "step9": 90},
        description="Target number of features for each step",
    )
    vif_threshold: float = Field(default=10.0, ge=1.0, le=100.0, description="VIF threshold for multicollinearity")
    correlation_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Correlation threshold")
    variance_threshold: float = Field(default=0.01, ge=0.0, le=1.0, description="Variance threshold")
    mutual_info_threshold: float = Field(default=0.001, ge=0.0, description="Mutual information threshold")
    shap_threshold: float = Field(default=0.001, ge=0.0, description="SHAP importance threshold")

    @validator("methods")
    def validate_methods(self, v):
        valid_methods = ["mutual_info", "lasso", "random_forest", "shap"]
        for method in v:
            if method not in valid_methods:
                msg = f"Invalid method '{method}'. Valid methods: {valid_methods}"
                raise ValueError(msg)
        return v


class RegularizationConfig(BaseModel):
    """Configuration for regularization optimization."""

    enable: bool = Field(default=True, description="Enable regularization optimization")
    l1_alpha_range: list[float] = Field(default=[0.001, 0.1], description="L1 alpha range")
    l2_alpha_range: list[float] = Field(default=[0.0001, 0.01], description="L2 alpha range")
    dropout_range: list[float] = Field(default=[0.1, 0.5], description="Dropout range")

    model_specific: dict[str, dict[str, Any]] = Field(
        default={
            "lightgbm": {
                "reg_alpha_range": [0.001, 0.1],
                "reg_lambda_range": [0.0001, 0.01],
            },
            "neural_networks": {
                "weight_decay_range": [1e-6, 1e-3],
                "dropout_range": [0.1, 0.5],
            },
        },
        description="Model-specific regularization parameters",
    )

    @validator("l1_alpha_range", "l2_alpha_range", "dropout_range")
    def validate_ranges(self, v):
        if len(v) != 2:
            msg = "Range must have exactly 2 values [min, max]"
            raise ValueError(msg)
        if v[0] >= v[1]:
            msg = "Range min must be less than max"
            raise ValueError(msg)
        return v


class OptunaConfig(BaseModel):
    """Configuration for Optuna hyperparameter optimization."""

    enable: bool = Field(default=True, description="Enable Optuna optimization")
    n_trials_per_batch: int = Field(default=50, ge=1, le=1000, description="Trials per batch")
    n_batches: int = Field(default=3, ge=1, le=10, description="Number of batches")
    timeout_per_batch: int = Field(default=300, ge=60, le=3600, description="Timeout per batch in seconds")
    sampler: SamplerType = Field(default=SamplerType.TPE, description="Optuna sampler")
    pruner: PrunerType = Field(default=PrunerType.MEDIAN, description="Optuna pruner")
    storage: str | None = Field(default=None, description="Optuna storage URL")

    @validator("timeout_per_batch")
    def validate_timeout(self, v):
        if v < 60:
            msg = "Timeout must be at least 60 seconds"
            raise ValueError(msg)
        return v


class VectorizationConfig(BaseModel):
    """Configuration for vectorized operations."""

    enable: bool = Field(default=True, description="Enable vectorized operations")
    batch_size: int = Field(default=1024, ge=32, le=10000, description="Batch size for operations")
    use_gpu: bool = Field(default=True, description="Use GPU if available")
    memory_efficient: bool = Field(default=True, description="Use memory-efficient operations")


class ExperimentTrackingConfig(BaseModel):
    """Configuration for experiment tracking."""

    enable: bool = Field(default=True, description="Enable experiment tracking")
    mlflow: bool = Field(default=True, description="Enable MLflow tracking")
    wandb: bool = Field(default=False, description="Enable Weights & Biases tracking")
    log_artifacts: bool = Field(default=True, description="Log model artifacts")
    log_metrics: bool = Field(default=True, description="Log detailed metrics")


class EnhancedLMOptimizerConfig(BaseModel):
    """Main configuration for Enhanced LM Optimizer."""

    feature_selection: FeatureSelectionConfig = Field(
        default_factory=FeatureSelectionConfig,
        description="Feature selection configuration",
    )
    regularization: RegularizationConfig = Field(
        default_factory=RegularizationConfig,
        description="Regularization configuration",
    )
    optuna: OptunaConfig = Field(
        default_factory=OptunaConfig,
        description="Optuna configuration",
    )
    vectorization: VectorizationConfig = Field(
        default_factory=VectorizationConfig,
        description="Vectorization configuration",
    )
    experiment_tracking: ExperimentTrackingConfig = Field(
        default_factory=ExperimentTrackingConfig,
        description="Experiment tracking configuration",
    )

    # Performance settings
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    max_workers: int = Field(default=4, ge=1, le=16, description="Maximum number of workers")
    cache_results: bool = Field(default=True, description="Cache optimization results")

    # Validation settings
    validate_data_quality: bool = Field(default=True, description="Validate data quality before optimization")
    check_memory_usage: bool = Field(default=True, description="Check memory usage during optimization")

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"  # Prevent additional fields
        json_encoders = {
            # Custom JSON encoders if needed
        }

    @classmethod
    def validate_config(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        # Check for potential issues
        if self.optuna.n_trials_per_batch * self.optuna.n_batches > 1000:
            warnings.append("Total trials > 1000 may take a long time to complete")

        if self.vectorization.batch_size > 2048:
            warnings.append("Large batch size may cause memory issues")

        if self.max_workers > 8:
            warnings.append("High number of workers may cause resource contention")

        return warnings


# Default configuration
DEFAULT_CONFIG = EnhancedLMOptimizerConfig()

# Configuration presets

