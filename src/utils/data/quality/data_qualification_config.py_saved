from src.utils.tprint import tprint

"""
Unified Configuration System for Data Qualification Pipeline

This module provides a centralized configuration management system for all data qualification steps,
ensuring consistent configuration handling, validation, and type safety.

Key Features:
- Unified configuration schema for all data qualification steps
- Comprehensive validation with detailed error messages
- Type-safe configuration access with proper defaults
- Environment-specific configuration support
- Configuration inheritance and composition
- Performance optimization settings
- ML Commons integration configuration
"""

import json
import yaml
from typing import Dict, Any, Optional, List, Union, Type, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import logging
from datetime import datetime
import os

from src.utils.ml_common.transaction_costs import DEFAULT_TRANSACTION_COST

# Initialize logger
logger = logging.getLogger(__name__)

T = TypeVar('T')

class RegimeDetectionMethod(Enum):
    """Enumeration for regime detection methods."""
    HMM_GAUSSIAN = "hmm_gaussian"
    HMM_MIXTURE = "hmm_mixture"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    KMEANS = "kmeans"
    ENSEMBLE = "ensemble"

class LabelingMethod(Enum):
    """Enumeration for labeling methods."""
    TRIPLE_BARRIER = "triple_barrier"
    REGIME_AWARE = "regime_aware"
    FRACTIONAL = "fractional"
    PROFIT_BASED = "profit_based"

class ProcessingMode(Enum):
    """Enumeration for processing modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"

@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validated_config: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    enable_m1_optimization: bool = True
    enable_gpu_acceleration: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    chunk_size: int = 1000
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_profiling: bool = False
    profiling_output_dir: Optional[str] = None

@dataclass
class SROptimizationConfig:
    """Support/Resistance optimization configuration."""
    min_touch_count: int = 3
    max_touch_count: int = 10
    touch_tolerance: float = 0.02
    strength_threshold: float = 0.5
    lookback_periods: List[int] = field(default_factory=lambda: [20, 50, 100])
    enable_dynamic_optimization: bool = True
    optimization_iterations: int = 100
    random_state: int = 42
    enable_ml_commons: bool = True
    enable_confidence_metrics: bool = True

@dataclass
class HMMRegimeConfig:
    """HMM regime detection configuration."""
    n_regimes: int = 3
    method: RegimeDetectionMethod = RegimeDetectionMethod.HMM_GAUSSIAN
    n_iterations: int = 100
    min_regime_duration: int = 5
    max_regime_duration: int = 1000
    min_regime_samples: int = 100
    max_regime_samples: int = 10000
    random_state: int = 42
    covariance_type: str = "full"
    enable_ensemble: bool = False
    ensemble_methods: List[RegimeDetectionMethod] = field(default_factory=lambda: [
        RegimeDetectionMethod.HMM_GAUSSIAN,
        RegimeDetectionMethod.GAUSSIAN_MIXTURE
    ])

@dataclass
class TripleBarrierConfig:
    """Triple barrier labeling configuration."""
    profit_take_multiplier: float = 0.02
    stop_loss_multiplier: float = 0.01
    time_barrier_minutes: int = 30
    max_lookahead: int = 100
    transaction_cost: float = DEFAULT_TRANSACTION_COST
    regime_aware: bool = True
    regime_column: str = "regime"
    enable_fractional_barriers: bool = False
    fractional_threshold: float = 0.5
    enable_ml_commons: bool = True

@dataclass
class RegimeProcessingConfig:
    """Regime data processing configuration."""
    min_regime_samples: int = 100
    max_regime_samples: int = 10000
    chunk_size: int = 1000
    memory_efficient: bool = True
    validate_continuity: bool = True
    enable_async_processing: bool = True
    max_concurrent_chunks: int = 4
    enable_data_type_optimization: bool = True
    enable_memory_pool: bool = True

@dataclass
class MLCommonsConfig:
    """ML Commons integration configuration."""
    enable_ml_commons: bool = True
    enable_fallback: bool = True
    data_quality_config: Dict[str, Any] = field(default_factory=lambda: {
        'outlier_contamination': 0.1,
        'missing_threshold': 0.3,
        'correlation_method': 'spearman'
    })
    pipeline_orchestrator_config: Dict[str, Any] = field(default_factory=lambda: {
        'max_workers': 4,
        'enable_parallel': True,
        'default_timeout': 1800
    })
    feature_selection_config: Dict[str, Any] = field(default_factory=lambda: {
        'enable_gpu': True,
        'enable_parallel': True,
        'max_workers': 4,
        'random_state': 42
    })
    parallel_processing_config: Dict[str, Any] = field(default_factory=lambda: {
        'max_workers': 4,
        'enable_joblib': True,
        'chunk_size': 5000
    })

@dataclass
class DataQualificationConfig:
    """
    Unified configuration for all data qualification steps.

    This configuration class provides a centralized way to manage all settings
    for the data qualification pipeline, with comprehensive validation and
    type safety.

    Example:
        >>> config = DataQualificationConfig(
        ...     symbol="AAPL",
        ...     exchange="NASDAQ",
        ...     timeframe="1m",
        ...     data_dir="/path/to/data"
        ... )
        >>> validation_result = config.validate()
        >>> if validation_result.is_valid:
        ...     tprint("Configuration is valid")
    """

    # Core identification
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str

    # Performance settings
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # Step-specific configurations
    sr_optimization: SROptimizationConfig = field(default_factory=SROptimizationConfig)
    hmm_regime: HMMRegimeConfig = field(default_factory=HMMRegimeConfig)
    triple_barrier: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    regime_processing: RegimeProcessingConfig = field(default_factory=RegimeProcessingConfig)

    # ML Commons integration
    ml_commons: MLCommonsConfig = field(default_factory=MLCommonsConfig)

    # Additional settings
    enable_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_metrics: bool = True
    metrics_output_dir: Optional[str] = None
    enable_debugging: bool = False
    debug_output_dir: Optional[str] = None

    # Environment settings
    environment: str = "development"
    config_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def validate(self) -> ValidationResult:
        """
        Validate the entire configuration.

        Returns:
            ValidationResult with validation status and any errors/warnings

        Example:
            >>> config = DataQualificationConfig(symbol="AAPL", exchange="NASDAQ", timeframe="1m", data_dir="/data")
            >>> result = config.validate()
            >>> if not result.is_valid:
            ...     tprint(f"Validation errors: {result.errors}")
        """
        errors = []
        warnings = []

        # Validate core fields
        if not self.symbol or not isinstance(self.symbol, str):
            errors.append("Symbol must be a non-empty string")

        if not self.exchange or not isinstance(self.exchange, str):
            errors.append("Exchange must be a non-empty string")

        if not self.timeframe or not isinstance(self.timeframe, str):
            errors.append("Timeframe must be a non-empty string")

        if not self.data_dir or not isinstance(self.data_dir, str):
            errors.append("Data directory must be a non-empty string")

        # Validate data directory exists
        if self.data_dir and not Path(self.data_dir).exists():
            warnings.append(f"Data directory does not exist: {self.data_dir}")

        # Validate performance configuration
        if self.performance.max_workers < 1:
            errors.append("Max workers must be at least 1")

        if self.performance.memory_limit_gb <= 0:
            errors.append("Memory limit must be positive")

        if self.performance.chunk_size < 1:
            errors.append("Chunk size must be at least 1")

        # Validate SR optimization configuration
        if self.sr_optimization.min_touch_count < 1:
            errors.append("Min touch count must be at least 1")

        if self.sr_optimization.max_touch_count < self.sr_optimization.min_touch_count:
            errors.append("Max touch count must be >= min touch count")

        if not (0 < self.sr_optimization.touch_tolerance < 1):
            errors.append("Touch tolerance must be between 0 and 1")

        if not (0 < self.sr_optimization.strength_threshold < 1):
            errors.append("Strength threshold must be between 0 and 1")

        # Validate HMM regime configuration
        if self.hmm_regime.n_regimes < 2:
            errors.append("Number of regimes must be at least 2")

        if self.hmm_regime.n_iterations < 1:
            errors.append("Number of iterations must be at least 1")

        if self.hmm_regime.min_regime_duration < 1:
            errors.append("Min regime duration must be at least 1")

        if self.hmm_regime.max_regime_duration < self.hmm_regime.min_regime_duration:
            errors.append("Max regime duration must be >= min regime duration")

        # Validate triple barrier configuration
        if self.triple_barrier.profit_take_multiplier <= 0:
            errors.append("Profit take multiplier must be positive")

        if self.triple_barrier.stop_loss_multiplier <= 0:
            errors.append("Stop loss multiplier must be positive")

        if self.triple_barrier.time_barrier_minutes < 1:
            errors.append("Time barrier must be at least 1 minute")

        if self.triple_barrier.max_lookahead < 1:
            errors.append("Max lookahead must be at least 1")

        if not (0 <= self.triple_barrier.transaction_cost < 1):
            errors.append("Transaction cost must be between 0 and 1")

        # Validate regime processing configuration
        if self.regime_processing.min_regime_samples < 1:
            errors.append("Min regime samples must be at least 1")

        if self.regime_processing.max_regime_samples < self.regime_processing.min_regime_samples:
            errors.append("Max regime samples must be >= min regime samples")

        if self.regime_processing.chunk_size < 1:
            errors.append("Chunk size must be at least 1")

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            errors.append(f"Log level must be one of: {valid_log_levels}")

        # Validate environment
        valid_environments = ["development", "staging", "production"]
        if self.environment not in valid_environments:
            errors.append(f"Environment must be one of: {valid_environments}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validated_config=asdict(self) if len(errors) == 0 else None
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, file_path: Optional[str] = None) -> str:
        """
        Convert configuration to JSON string or save to file.

        Args:
            file_path: Optional file path to save JSON to

        Returns:
            JSON string representation of configuration

        Example:
            >>> config = DataQualificationConfig(symbol="AAPL", exchange="NASDAQ", timeframe="1m", data_dir="/data")
            >>> json_str = config.to_json()
            >>> config.to_json("config.json")  # Save to file
        """
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, default=str)

        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(json_str)
            logger.info(f"Configuration saved to {file_path}")

        return json_str

    def to_yaml(self, file_path: Optional[str] = None) -> str:
        """
        Convert configuration to YAML string or save to file.

        Args:
            file_path: Optional file path to save YAML to

        Returns:
            YAML string representation of configuration
        """
        config_dict = self.to_dict()
        yaml_str = yaml.dump(config_dict, default_flow_style=False, indent=2)

        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(yaml_str)
            logger.info(f"Configuration saved to {file_path}")

        return yaml_str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataQualificationConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration values

        Returns:
            DataQualificationConfig instance

        Example:
            >>> config_dict = {"symbol": "AAPL", "exchange": "NASDAQ", "timeframe": "1m", "data_dir": "/data"}
            >>> config = DataQualificationConfig.from_dict(config_dict)
        """
        # Handle nested configurations
        if 'performance' in config_dict and isinstance(config_dict['performance'], dict):
            config_dict['performance'] = PerformanceConfig(**config_dict['performance'])

        if 'sr_optimization' in config_dict and isinstance(config_dict['sr_optimization'], dict):
            config_dict['sr_optimization'] = SROptimizationConfig(**config_dict['sr_optimization'])

        if 'hmm_regime' in config_dict and isinstance(config_dict['hmm_regime'], dict):
            hmm_config = config_dict['hmm_regime']
            if 'method' in hmm_config and isinstance(hmm_config['method'], str):
                hmm_config['method'] = RegimeDetectionMethod(hmm_config['method'])
            config_dict['hmm_regime'] = HMMRegimeConfig(**hmm_config)

        if 'triple_barrier' in config_dict and isinstance(config_dict['triple_barrier'], dict):
            config_dict['triple_barrier'] = TripleBarrierConfig(**config_dict['triple_barrier'])

        if 'regime_processing' in config_dict and isinstance(config_dict['regime_processing'], dict):
            config_dict['regime_processing'] = RegimeProcessingConfig(**config_dict['regime_processing'])

        if 'ml_commons' in config_dict and isinstance(config_dict['ml_commons'], dict):
            config_dict['ml_commons'] = MLCommonsConfig(**config_dict['ml_commons'])

        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_str: str) -> 'DataQualificationConfig':
        """
        Create configuration from JSON string.

        Args:
            json_str: JSON string containing configuration

        Returns:
            DataQualificationConfig instance
        """
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json_file(cls, file_path: str) -> 'DataQualificationConfig':
        """
        Create configuration from JSON file.

        Args:
            file_path: Path to JSON configuration file

        Returns:
            DataQualificationConfig instance
        """
        with open(file_path, 'r') as f:
            json_str = f.read()
        return cls.from_json(json_str)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'DataQualificationConfig':
        """
        Create configuration from YAML string.

        Args:
            yaml_str: YAML string containing configuration

        Returns:
            DataQualificationConfig instance
        """
        config_dict = yaml.safe_load(yaml_str)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml_file(cls, file_path: str) -> 'DataQualificationConfig':
        """
        Create configuration from YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            DataQualificationConfig instance
        """
        with open(file_path, 'r') as f:
            yaml_str = f.read()
        return cls.from_yaml(yaml_str)

    @classmethod
    def from_environment(cls, prefix: str = "DQ_") -> 'DataQualificationConfig':
        """
        Create configuration from environment variables.

        Args:
            prefix: Prefix for environment variables (default: "DQ_")

        Returns:
            DataQualificationConfig instance with values from environment

        Example:
            >>> # Set environment variables: DQ_SYMBOL=AAPL, DQ_EXCHANGE=NASDAQ, etc.
            >>> config = DataQualificationConfig.from_environment()
        """
        config_dict = {}

        # Core fields
        config_dict['symbol'] = os.getenv(f"{prefix}SYMBOL", "")
        config_dict['exchange'] = os.getenv(f"{prefix}EXCHANGE", "")
        config_dict['timeframe'] = os.getenv(f"{prefix}TIMEFRAME", "1m")
        config_dict['data_dir'] = os.getenv(f"{prefix}DATA_DIR", "./data")

        # Performance settings
        performance_config = {}
        performance_config['enable_m1_optimization'] = os.getenv(f"{prefix}ENABLE_M1_OPTIMIZATION", "true").lower() == "true"
        performance_config['enable_gpu_acceleration'] = os.getenv(f"{prefix}ENABLE_GPU_ACCELERATION", "true").lower() == "true"
        performance_config['max_workers'] = int(os.getenv(f"{prefix}MAX_WORKERS", "4"))
        performance_config['memory_limit_gb'] = float(os.getenv(f"{prefix}MEMORY_LIMIT_GB", "8.0"))
        config_dict['performance'] = performance_config

        # Additional settings
        config_dict['enable_logging'] = os.getenv(f"{prefix}ENABLE_LOGGING", "true").lower() == "true"
        config_dict['log_level'] = os.getenv(f"{prefix}LOG_LEVEL", "INFO")
        config_dict['environment'] = os.getenv(f"{prefix}ENVIRONMENT", "development")

        return cls.from_dict(config_dict)

    def merge(self, other: 'DataQualificationConfig') -> 'DataQualificationConfig':
        """
        Merge this configuration with another configuration.

        Args:
            other: Another DataQualificationConfig to merge with

        Returns:
            New DataQualificationConfig with merged values

        Example:
            >>> base_config = DataQualificationConfig(symbol="AAPL", exchange="NASDAQ", timeframe="1m", data_dir="/data")
            >>> override_config = DataQualificationConfig(symbol="MSFT", exchange="NASDAQ", timeframe="1m", data_dir="/data")
            >>> merged_config = base_config.merge(override_config)
        """
        # Convert both to dictionaries
        self_dict = self.to_dict()
        other_dict = other.to_dict()

        # Merge dictionaries (other takes precedence)
        merged_dict = {**self_dict, **other_dict}

        # Handle nested configurations
        for key in ['performance', 'sr_optimization', 'hmm_regime', 'triple_barrier', 'regime_processing', 'ml_commons']:
            if key in other_dict and isinstance(other_dict[key], dict):
                if key in self_dict and isinstance(self_dict[key], dict):
                    merged_dict[key] = {**self_dict[key], **other_dict[key]}
                else:
                    merged_dict[key] = other_dict[key]

        return self.from_dict(merged_dict)

    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific step.

        Args:
            step_name: Name of the step (e.g., 'sr_optimization', 'hmm_regime')

        Returns:
            Dictionary containing step-specific configuration

        Example:
            >>> config = DataQualificationConfig(symbol="AAPL", exchange="NASDAQ", timeframe="1m", data_dir="/data")
            >>> sr_config = config.get_step_config('sr_optimization')
        """
        step_configs = {
            'sr_optimization': asdict(self.sr_optimization),
            'hmm_regime': asdict(self.hmm_regime),
            'triple_barrier': asdict(self.triple_barrier),
            'regime_processing': asdict(self.regime_processing),
            'performance': asdict(self.performance),
            'ml_commons': asdict(self.ml_commons)
        }

        if step_name not in step_configs:
            raise ValueError(f"Unknown step name: {step_name}. Available steps: {list(step_configs.keys())}")

        return step_configs[step_name]

    def update_step_config(self, step_name: str, updates: Dict[str, Any]) -> 'DataQualificationConfig':
        """
        Update configuration for a specific step.

        Args:
            step_name: Name of the step to update
            updates: Dictionary containing updates to apply

        Returns:
            New DataQualificationConfig with updated step configuration

        Example:
            >>> config = DataQualificationConfig(symbol="AAPL", exchange="NASDAQ", timeframe="1m", data_dir="/data")
            >>> updated_config = config.update_step_config('sr_optimization', {'min_touch_count': 5})
        """
        config_dict = self.to_dict()

        if step_name not in config_dict:
            raise ValueError(f"Unknown step name: {step_name}")

        # Update the step configuration
        if isinstance(config_dict[step_name], dict):
            config_dict[step_name].update(updates)
        else:
            config_dict[step_name] = updates

        return self.from_dict(config_dict)

class DataQualificationConfigManager:
    """
    Manager for data qualification configurations.

    Provides centralized management of configurations with support for
    multiple environments, configuration inheritance, and validation.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("./configs")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.getChild('ConfigManager')
        self._config_cache: Dict[str, DataQualificationConfig] = {}

    def load_config(
        self,
        config_name: str,
        environment: str = "development",
        validate: bool = True
    ) -> DataQualificationConfig:
        """
        Load configuration by name and environment.

        Args:
            config_name: Name of the configuration
            environment: Environment (development, staging, production)
            validate: Whether to validate the configuration

        Returns:
            DataQualificationConfig instance
        """
        cache_key = f"{config_name}_{environment}"

        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        # Try to load base configuration
        base_config_path = self.config_dir / f"{config_name}.json"
        if not base_config_path.exists():
            base_config_path = self.config_dir / f"{config_name}.yaml"

        if not base_config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {base_config_path}")

        # Load base configuration
        if base_config_path.suffix == '.json':
            config = DataQualificationConfig.from_json_file(str(base_config_path))
        else:
            config = DataQualificationConfig.from_yaml_file(str(base_config_path))

        # Try to load environment-specific overrides
        env_config_path = self.config_dir / f"{config_name}_{environment}.json"
        if not env_config_path.exists():
            env_config_path = self.config_dir / f"{config_name}_{environment}.yaml"

        if env_config_path.exists():
            if env_config_path.suffix == '.json':
                env_config = DataQualificationConfig.from_json_file(str(env_config_path))
            else:
                env_config = DataQualificationConfig.from_yaml_file(str(env_config_path))

            # Merge configurations
            config = config.merge(env_config)

        # Validate if requested
        if validate:
            validation_result = config.validate()
            if not validation_result.is_valid:
                raise ValueError(f"Configuration validation failed: {validation_result.errors}")

        # Cache the configuration
        self._config_cache[cache_key] = config

        self.logger.info(f"Loaded configuration: {config_name} (environment: {environment})")
        return config

    def save_config(
        self,
        config: DataQualificationConfig,
        config_name: str,
        environment: str = "development",
        format: str = "json"
    ) -> str:
        """
        Save configuration to file.

        Args:
            config: Configuration to save
            config_name: Name for the configuration file
            environment: Environment for the configuration
            format: File format (json or yaml)

        Returns:
            Path to the saved configuration file
        """
        filename = f"{config_name}_{environment}.{format}"
        file_path = self.config_dir / filename

        if format == "json":
            config.to_json(str(file_path))
        else:
            config.to_yaml(str(file_path))

        self.logger.info(f"Saved configuration: {file_path}")
        return str(file_path)

    def list_configs(self) -> List[str]:
        """List available configuration names."""
        config_files = list(self.config_dir.glob("*.json")) + list(self.config_dir.glob("*.yaml"))
        config_names = set()

        for file_path in config_files:
            name = file_path.stem
            # Remove environment suffix if present
            if '_' in name:
                name = '_'.join(name.split('_')[:-1])
            config_names.add(name)

        return sorted(list(config_names))

    def clear_cache(self):
        """Clear the configuration cache."""
        self._config_cache.clear()
        self.logger.info("Configuration cache cleared")

# Global configuration manager instance
_config_manager: Optional[DataQualificationConfigManager] = None

def get_config_manager() -> DataQualificationConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = DataQualificationConfigManager()
    return _config_manager

# Convenience functions
def create_default_config(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_dir: str = "./data"
) -> DataQualificationConfig:
    """
    Create a default configuration with common settings.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Data timeframe
        data_dir: Data directory path

    Returns:
        DataQualificationConfig with default settings
    """
    return DataQualificationConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir
    )

def load_config_from_file(file_path: str) -> DataQualificationConfig:
    """
    Load configuration from file (auto-detect format).

    Args:
        file_path: Path to configuration file

    Returns:
        DataQualificationConfig instance
    """
    file_path = Path(file_path)

    if file_path.suffix == '.json':
        return DataQualificationConfig.from_json_file(str(file_path))
    elif file_path.suffix in ['.yaml', '.yml']:
        return DataQualificationConfig.from_yaml_file(str(file_path))
    else:
        raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")

# Export main classes and functions
__all__ = [
    'DataQualificationConfig',
    'PerformanceConfig',
    'SROptimizationConfig',
    'HMMRegimeConfig',
    'TripleBarrierConfig',
    'RegimeProcessingConfig',
    'MLCommonsConfig',
    'ValidationResult',
    'RegimeDetectionMethod',
    'LabelingMethod',
    'ProcessingMode',
    'DataQualificationConfigManager',
    'get_config_manager',
    'create_default_config',
    'load_config_from_file'
]
