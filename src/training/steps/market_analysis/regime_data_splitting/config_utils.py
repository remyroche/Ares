"""
Configuration utilities for regime data splitting module.

This module provides centralized configuration management and path utilities
to eliminate hard-coded paths and improve maintainability.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import os


@dataclass
class RegimeDataSplittingConfig:
    """Configuration for regime data splitting operations."""
    
    # Data paths
    base_data_dir: str = "data"
    historical_data_dir: str = "historical_data"
    training_data_dir: str = "training"
    models_dir: str = "models"
    artifacts_dir: str = "artifacts"
    cache_dir: str = "data_cache"
    
    # HMM paths
    hmm_regimes_dir: str = "hmm_regimes"
    hmm_models_dir: str = "hmm"
    
    # File patterns
    market_data_pattern: str = "{exchange}_{symbol}_{timeframe}_market_data.parquet"
    regime_tagged_data_pattern: str = "{exchange}_{symbol}_{timeframe}_regime_tagged_data.parquet"
    unified_regime_data_pattern: str = "{exchange}_{symbol}_{timeframe}_unified_regime_data.parquet"
    composite_clusters_pattern: str = "{exchange}_{symbol}_{timeframe}_composite_clusters.parquet"
    regime_statistics_pattern: str = "regime_statistics.json"
    
    # Processing parameters
    min_rows: int = 100
    max_memory_gb: float = 8.0
    chunk_size: int = 100_000
    max_concurrent_files: int = 3
    
    # Validation parameters
    min_regimes: int = 2
    max_regimes: int = 20
    data_quality_threshold: float = 0.7
    
    # HMM parameters
    n_features: int = 100
    enable_hmm_models: bool = True
    use_ensemble_models: bool = True
    
    # Timeout and retry settings
    file_operation_timeout: int = 300  # seconds
    max_retries: int = 3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.min_regimes < 1:
            raise ValueError("min_regimes must be at least 1")
        if self.max_regimes <= self.min_regimes:
            raise ValueError("max_regimes must be greater than min_regimes")
        if not 0 < self.data_quality_threshold <= 1:
            raise ValueError("data_quality_threshold must be between 0 and 1")


class PathManager:
    """Centralized path management for regime data splitting."""
    
    def __init__(self, config: RegimeDataSplittingConfig):
        self.config = config
    
    def get_base_data_dir(self) -> Path:
        """Get base data directory path."""
        return Path(self.config.base_data_dir)
    
    def get_historical_data_dir(self) -> Path:
        """Get historical data directory path."""
        return Path(self.config.historical_data_dir)
    
    def get_training_data_dir(self, data_dir: Optional[str] = None) -> Path:
        """Get training data directory path."""
        base_dir = Path(data_dir) if data_dir else self.get_base_data_dir()
        return base_dir / self.config.training_data_dir
    
    def get_models_dir(self, data_dir: Optional[str] = None) -> Path:
        """Get models directory path."""
        base_dir = Path(data_dir) if data_dir else self.get_base_data_dir()
        return base_dir / self.config.models_dir
    
    def get_artifacts_dir(self) -> Path:
        """Get artifacts directory path."""
        return Path(self.config.artifacts_dir)
    
    def get_hmm_regimes_dir(self, data_dir: Optional[str] = None) -> Path:
        """Get HMM regimes directory path."""
        base_dir = Path(data_dir) if data_dir else self.get_base_data_dir()
        return base_dir / self.config.hmm_regimes_dir
    
    def get_hmm_models_dir(self, data_dir: Optional[str] = None) -> Path:
        """Get HMM models directory path."""
        models_dir = self.get_models_dir(data_dir)
        return models_dir / self.config.hmm_models_dir
    
    def get_regime_splits_dir(self, exchange: str, symbol: str, data_dir: Optional[str] = None) -> Path:
        """Get regime splits directory path."""
        historical_dir = Path(data_dir) if data_dir else self.get_historical_data_dir()
        return historical_dir / exchange.lower() / symbol.lower() / "regime_splits"
    
    def get_market_data_path(self, exchange: str, symbol: str, timeframe: str, data_dir: Optional[str] = None) -> Path:
        """Get market data file path."""
        training_dir = self.get_training_data_dir(data_dir)
        filename = self.config.market_data_pattern.format(
            exchange=exchange, symbol=symbol, timeframe=timeframe
        )
        return training_dir / filename
    
    def get_regime_tagged_data_path(self, exchange: str, symbol: str, timeframe: str, data_dir: Optional[str] = None) -> Path:
        """Get regime tagged data file path."""
        training_dir = self.get_training_data_dir(data_dir)
        filename = self.config.regime_tagged_data_pattern.format(
            exchange=exchange, symbol=symbol, timeframe=timeframe
        )
        return training_dir / filename
    
    def get_unified_regime_data_path(self, exchange: str, symbol: str, timeframe: str, data_dir: Optional[str] = None) -> Path:
        """Get unified regime data file path."""
        regime_splits_dir = self.get_regime_splits_dir(exchange, symbol, data_dir)
        filename = self.config.unified_regime_data_pattern.format(
            exchange=exchange, symbol=symbol, timeframe=timeframe
        )
        return regime_splits_dir / filename
    
    def get_composite_clusters_path(self, exchange: str, symbol: str, timeframe: str, data_dir: Optional[str] = None) -> Path:
        """Get composite clusters file path."""
        hmm_regimes_dir = self.get_hmm_regimes_dir(data_dir)
        filename = self.config.composite_clusters_pattern.format(
            exchange=exchange, symbol=symbol, timeframe=timeframe
        )
        return hmm_regimes_dir / filename
    
    def get_regime_statistics_path(self, exchange: str, symbol: str, data_dir: Optional[str] = None) -> Path:
        """Get regime statistics file path."""
        models_dir = self.get_models_dir(data_dir)
        return models_dir / exchange.lower() / symbol.lower() / self.config.regime_statistics_pattern
    
    def get_hmm_base_model_path(self, model_name: str, exchange: str, symbol: str, timeframe: str, data_dir: Optional[str] = None) -> Path:
        """Get HMM base model file path."""
        hmm_models_dir = self.get_hmm_models_dir(data_dir)
        base_models_dir = hmm_models_dir / "base_models"
        filename = f"hmm_base_{model_name}_{symbol}_{exchange}_{timeframe}.pkl"
        return base_models_dir / filename
    
    def get_hmm_ensemble_model_path(self, model_name: str, exchange: str, symbol: str, timeframe: str, data_dir: Optional[str] = None) -> Path:
        """Get HMM ensemble model file path."""
        hmm_models_dir = self.get_hmm_models_dir(data_dir)
        ensemble_models_dir = hmm_models_dir / "ensemble_models"
        filename = f"hmm_ensemble_{model_name}_{symbol}_{exchange}_{timeframe}.pkl"
        return ensemble_models_dir / filename
    
    def ensure_directories_exist(self, *paths: Path) -> None:
        """Ensure directories exist, creating them if necessary."""
        for path in paths:
            if path.suffix:  # It's a file path
                path.parent.mkdir(parents=True, exist_ok=True)
            else:  # It's a directory path
                path.mkdir(parents=True, exist_ok=True)
    
    def get_artifact_path(self, artifact_name: str, symbol: str, exchange: str, timeframe: str) -> Path:
        """Get artifact file path."""
        artifacts_dir = self.get_artifacts_dir()
        filename = f"{artifact_name}_{symbol}_{exchange}_{timeframe}.json"
        return artifacts_dir / "regime_data_splitting" / filename


class ConfigManager:
    """Configuration manager for regime data splitting."""
    
    def __init__(self, config: Optional[RegimeDataSplittingConfig] = None):
        self.config = config or RegimeDataSplittingConfig()
        self.path_manager = PathManager(self.config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigManager':
        """Create config manager from dictionary."""
        config = RegimeDataSplittingConfig(**config_dict)
        return cls(config)
    
    @classmethod
    def from_env(cls) -> 'ConfigManager':
        """Create config manager from environment variables."""
        config_dict = {}
        
        # Map environment variables to config fields
        env_mapping = {
            'REGIME_BASE_DATA_DIR': 'base_data_dir',
            'REGIME_HISTORICAL_DATA_DIR': 'historical_data_dir',
            'REGIME_MAX_MEMORY_GB': 'max_memory_gb',
            'REGIME_CHUNK_SIZE': 'chunk_size',
            'REGIME_MIN_REGIMES': 'min_regimes',
            'REGIME_MAX_REGIMES': 'max_regimes',
            'REGIME_DATA_QUALITY_THRESHOLD': 'data_quality_threshold',
        }
        
        for env_var, config_field in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Convert to appropriate type
                if config_field in ['max_memory_gb', 'data_quality_threshold']:
                    config_dict[config_field] = float(value)
                elif config_field in ['chunk_size', 'min_regimes', 'max_regimes']:
                    config_dict[config_field] = int(value)
                else:
                    config_dict[config_field] = value
        
        return cls.from_dict(config_dict)
    
    def get_config(self) -> RegimeDataSplittingConfig:
        """Get the configuration object."""
        return self.config
    
    def get_path_manager(self) -> PathManager:
        """Get the path manager."""
        return self.path_manager
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self.config, field.name)
            for field in self.config.__dataclass_fields__.values()
        }


# Global configuration manager instance
_global_config_manager = None


def get_config_manager(config: Optional[Union[Dict[str, Any], RegimeDataSplittingConfig]] = None) -> ConfigManager:
    """Get the global configuration manager instance."""
    global _global_config_manager
    
    if _global_config_manager is None:
        if isinstance(config, dict):
            _global_config_manager = ConfigManager.from_dict(config)
        elif isinstance(config, RegimeDataSplittingConfig):
            _global_config_manager = ConfigManager(config)
        else:
            _global_config_manager = ConfigManager()
    
    return _global_config_manager


def get_path_manager(config: Optional[Union[Dict[str, Any], RegimeDataSplittingConfig]] = None) -> PathManager:
    """Get the path manager instance."""
    config_manager = get_config_manager(config)
    return config_manager.get_path_manager()


def reset_global_config():
    """Reset the global configuration manager (useful for testing)."""
    global _global_config_manager
    _global_config_manager = None