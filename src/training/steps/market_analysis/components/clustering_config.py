"""
Configuration management for HDBSCAN clustering components.

This module provides centralized configuration management with validation,
inheritance, and documentation for clustering-related components.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import os

from src.utils.tprint import tprint_info, tprint_warning, tprint_error
from ..shared_utils import BaseConfig, ConfigValidator
from ..shared_utils.calibration_registry import get_quality_thresholds

@dataclass
class ClusteringConfig(BaseConfig):
    """Base configuration for clustering components with enhanced validation."""

    # Core clustering parameters
    n_regimes: int = 8
    algorithm_type: str = "adaptive_clustering"
    enable_economic_clustering: bool = True
    enable_ensemble_clustering: bool = True

    # Regime search bounds
    regime_search_min: int = 5
    regime_search_max: int = 15

    # Feature configuration
    feature_categories: List[str] = field(default_factory=lambda: [
        'regime_volatility',
        'regime_volume',
        'regime_structural_trend',
        'regime_statistical'
    ])
    use_regime_focused_features: bool = True
    exclude_trading_features: bool = True
    use_standardized_features: bool = True

    # Regime-specific weights
    economic_weight: float = 0.25
    volatility_regime_weight: float = 0.30
    volume_regime_weight: float = 0.25
    structural_trend_weight: float = 0.20

    # Quality thresholds (calibrated dynamically)
    min_regime_persistence: Optional[float] = None
    max_feature_noise_ratio: Optional[float] = None
    min_temporal_stability: Optional[float] = None

    # Output configuration
    output_dir: str = "data_cache"
    save_intermediate_results: bool = True

    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Validate regime search bounds
        self.regime_search_min = max(3, min(20, self.regime_search_min))
        self.regime_search_max = max(
            self.regime_search_min,
            min(25, self.regime_search_max),
        )

        # Ensure n_regimes is within bounds
        if not (self.regime_search_min <= self.n_regimes <= self.regime_search_max):
            self.n_regimes = max(
                self.regime_search_min,
                min(self.regime_search_max, self.n_regimes),
            )
            tprint_warning(f"Adjusted n_regimes to {self.n_regimes} to fit within bounds")

        # Validate weights sum to 1.0
        total_weight = (
            self.economic_weight +
            self.volatility_regime_weight +
            self.volume_regime_weight +
            self.structural_trend_weight
        )

        if abs(total_weight - 1.0) > 0.01:
            tprint_warning(f"Weights sum to {total_weight:.3f}, normalizing to 1.0")
            self.economic_weight /= total_weight
            self.volatility_regime_weight /= total_weight
            self.volume_regime_weight /= total_weight
            self.structural_trend_weight /= total_weight

        # Apply calibrated quality thresholds
        self._apply_calibrated_thresholds()

        # Validate output directory
        self._validate_output_directory()

        # Call parent validation
        super().__post_init__()

    def _apply_calibrated_thresholds(self) -> None:
        """Apply calibrated quality thresholds if not explicitly provided."""
        try:
            thresholds = get_quality_thresholds()

            if self.min_regime_persistence is None:
                self.min_regime_persistence = thresholds.get('min_regime_persistence', 0.7)

            if self.max_feature_noise_ratio is None:
                self.max_feature_noise_ratio = thresholds.get('max_feature_noise_ratio', 0.3)

            if self.min_temporal_stability is None:
                self.min_temporal_stability = thresholds.get('min_temporal_stability', 0.6)

        except Exception as e:
            tprint_warning(f"Failed to load calibrated thresholds: {e}")
            # Use defaults
            if self.min_regime_persistence is None:
                self.min_regime_persistence = 0.7
            if self.max_feature_noise_ratio is None:
                self.max_feature_noise_ratio = 0.3
            if self.min_temporal_stability is None:
                self.min_temporal_stability = 0.6

    def _validate_output_directory(self) -> None:
        """Validate and create output directory if needed."""
        try:
            output_path = Path(self.output_dir)
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
                tprint_info(f"Created output directory: {output_path}")

            # Check write permissions
            if not os.access(output_path, os.W_OK):
                raise PermissionError(f"No write access to output directory: {output_path}")

        except Exception as e:
            tprint_error(f"Output directory validation failed: {e}")
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'n_regimes': self.n_regimes,
            'algorithm_type': self.algorithm_type,
            'enable_economic_clustering': self.enable_economic_clustering,
            'enable_ensemble_clustering': self.enable_ensemble_clustering,
            'regime_search_min': self.regime_search_min,
            'regime_search_max': self.regime_search_max,
            'feature_categories': self.feature_categories,
            'use_regime_focused_features': self.use_regime_focused_features,
            'exclude_trading_features': self.exclude_trading_features,
            'use_standardized_features': self.use_standardized_features,
            'economic_weight': self.economic_weight,
            'volatility_regime_weight': self.volatility_regime_weight,
            'volume_regime_weight': self.volume_regime_weight,
            'structural_trend_weight': self.structural_trend_weight,
            'min_regime_persistence': self.min_regime_persistence,
            'max_feature_noise_ratio': self.max_feature_noise_ratio,
            'min_temporal_stability': self.min_temporal_stability,
            'output_dir': self.output_dir,
            'save_intermediate_results': self.save_intermediate_results,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClusteringConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)

            tprint_info(f"Configuration saved to {file_path}")

        except Exception as e:
            tprint_error(f"Failed to save configuration: {e}")
            raise

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'ClusteringConfig':
        """Load configuration from JSON file."""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {file_path}")

            with open(file_path, 'r') as f:
                config_dict = json.load(f)

            return cls.from_dict(config_dict)

        except Exception as e:
            tprint_error(f"Failed to load configuration: {e}")
            raise

@dataclass
class NASTASClusteringConfig(ClusteringConfig):
    """Configuration specifically for HDBSCAN clustering component."""

    exchange: str = "binance"

    # HDBSCAN specific parameters
    enable_label_fusion: bool = True
    fusion_method: str = "dawid_skene"
    fusion_max_iterations: int = 50
    fusion_tolerance: float = 1e-6

    # Cross-validation parameters
    cv_folds: int = 5
    cv_strategy: str = "time_series"
    enable_purged_cv: bool = True

    # Performance optimization
    enable_m1_optimization: bool = True
    enable_gpu_acceleration: bool = True
    memory_limit_mb: Optional[int] = None

    def __post_init__(self):
        """Validate HDBSCAN specific configuration."""
        super().__post_init__()

        # Validate fusion parameters
        if self.fusion_max_iterations < 10:
            self.fusion_max_iterations = 10
            tprint_warning("fusion_max_iterations too low, set to 10")

        if self.fusion_tolerance <= 0:
            self.fusion_tolerance = 1e-6
            tprint_warning("fusion_tolerance must be positive, set to 1e-6")

        # Validate CV parameters
        if self.cv_folds < 2:
            self.cv_folds = 2
            tprint_warning("cv_folds too low, set to 2")

        # Validate memory limit
        if self.memory_limit_mb is not None and self.memory_limit_mb < 100:
            self.memory_limit_mb = 100
            tprint_warning("memory_limit_mb too low, set to 100")

class ConfigurationManager:
    """Centralized configuration management with validation and persistence."""

    def __init__(self, default_config: Optional[ClusteringConfig] = None):
        """Initialize configuration manager."""
        self.default_config = default_config or ClusteringConfig()
        self.validator = ConfigValidator(verbose=True)

    def create_config(
        self,
        config_type: str = "clustering",
        **kwargs
    ) -> Union[ClusteringConfig, NASTASClusteringConfig]:
        """Create configuration with validation."""
        try:
            if config_type == "nas_tas":
                config = NASTASClusteringConfig(**kwargs)
            else:
                config = ClusteringConfig(**kwargs)

            # Validate configuration
            self.validator.validate_config(config)

            tprint_info(f"Created {config_type} configuration with {len(kwargs)} parameters")
            return config

        except Exception as e:
            tprint_error(f"Failed to create configuration: {e}")
            raise

    def validate_config(self, config: Union[ClusteringConfig, NASTASClusteringConfig]) -> bool:
        """Validate configuration and return success status."""
        try:
            self.validator.validate_config(config)
            tprint_info("Configuration validation passed")
            return True

        except Exception as e:
            tprint_error(f"Configuration validation failed: {e}")
            return False

    def merge_configs(
        self,
        base_config: Union[ClusteringConfig, NASTASClusteringConfig],
        override_config: Dict[str, Any]
    ) -> Union[ClusteringConfig, NASTASClusteringConfig]:
        """Merge base configuration with overrides."""
        try:
            base_dict = base_config.to_dict()
            base_dict.update(override_config)

            if isinstance(base_config, NASTASClusteringConfig):
                return NASTASClusteringConfig.from_dict(base_dict)
            else:
                return ClusteringConfig.from_dict(base_dict)

        except Exception as e:
            tprint_error(f"Failed to merge configurations: {e}")
            raise
