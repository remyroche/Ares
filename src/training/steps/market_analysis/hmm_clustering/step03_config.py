from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
#!/usr/bin/env python3
"""Step03 Configuration Management.

Centralized configuration for HMM regime discovery with all parameters
organized by component and use case.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import json
from pathlib import Path
import logging


@dataclass
class BayesianOptimizationConfig:
    """Configuration for Bayesian parameter optimization."""
    n_trials: int = 100
    timeout_minutes: int = 30
    cv_folds: int = 3
    random_state: int = 42
    n_startup_trials: int = 5
    n_warmup_steps: int = 10
    pruner_type: str = "median"
    sampler_type: str = "tpe"


@dataclass
class HMMConfig:
    """Configuration for HMM clustering."""
    n_components_range: List[int] = field(default_factory = lambda: [2, 40])
    covariance_types: List[str] = field(default_factory = lambda: ["full", "tied", "diag", "spherical"])
    n_iter_range: List[int] = field(default_factory = lambda: [50, 200])
    tol_range: List[float] = field(default_factory = lambda: [1e-6, 1e-2])
    reg_covar_range: List[float] = field(default_factory = lambda: [1e-7, 1e-2])
    max_samples: int = 5000
    random_state: int = 42


@dataclass
class EnsembleConfig:
    """Configuration for ensemble clustering."""
    weights: Dict[str, float] = field(default_factory = lambda: {"hmm": 0.4, "kmeans": 0.3, "dbscan": 0.3})
    kmeans_n_clusters_range: List[int] = field(default_factory = lambda: [10, 30])
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    dbscan_eps_range: List[float] = field(default_factory = lambda: [0.1, 2.0])
    dbscan_min_samples_range: List[int] = field(default_factory = lambda: [5, 50])
    n_jobs: int = -1
    random_state: int = 42


@dataclass
class EconomicValidationConfig:
    """Configuration for economic significance validation."""
    significance_threshold: float = 0.05
    economic_threshold: float = 0.001
    min_regime_size: int = 20
    min_regime_duration: int = 10
    annualization_factor: int = 252


@dataclass
class MLTransitionConfig:
    """Configuration for ML transition detection."""
    initial_features: int = 20
    feature_increment: int = 10
    max_features: int = 100
    min_improvement: float = 0.001
    patience: int = 3
    prediction_horizon: int = 5
    random_state: int = 42
    
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_class_weight: str = "balanced"
    rf_n_jobs: int = -1
    
    # LGBM parameters
    lgb_objective: str = "binary"
    lgb_metric: str = "binary_logloss"
    lgb_boosting_type: str = "gbdt"
    lgb_num_leaves: int = 31
    lgb_learning_rate: float = 0.05
    lgb_feature_fraction: float = 0.9
    lgb_bagging_fraction: float = 0.8
    lgb_bagging_freq: int = 5
    lgb_verbose: int = -1


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""
    # Price features
    momentum_windows: List[int] = field(default_factory = lambda: [1, 2, 3, 5, 10, 20])
    price_position_windows: List[int] = field(default_factory = lambda: [10, 20, 50])
    volatility_windows: List[int] = field(default_factory = lambda: [5, 10, 20])
    
    # Volume features
    volume_momentum_windows: List[int] = field(default_factory = lambda: [1, 2, 3, 5, 10, 20])
    volume_ratio_windows: List[int] = field(default_factory = lambda: [5, 10, 20, 50])
    
    # Technical indicators
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    atr_window: int = 14
    adx_window: int = 14
    bb_window: int = 20
    bb_std: float = 2.0
    
    # Moving averages
    sma_windows: List[int] = field(default_factory = lambda: [20, 50])
    ema_spans: List[int] = field(default_factory = lambda: [12, 26])
    
    # Lagged features
    lag_periods: List[int] = field(default_factory = lambda: [1, 2, 3, 5, 10])
    
    # Feature processing
    max_features_before_pca: int = 50
    pca_components: Optional[int] = None
    fill_method: str = "zero"  # "zero", "forward", "backward", "interpolate"


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    chunk_size: int = 10000
    max_memory_usage_gb: float = 8.0
    enable_garbage_collection: bool = True
    gc_frequency: int = 1000  # Run GC every N operations
    use_memory_mapping: bool = False
    temp_dir: str = "temp"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    max_log_size_mb: int = 100
    backup_count: int = 5


@dataclass
class Step03Config:
    """Main configuration class for Step03."""
    
    # Component configurations
    bayesian_optimization: BayesianOptimizationConfig = field(default_factory = BayesianOptimizationConfig)
    hmm: HMMConfig = field(default_factory = HMMConfig)
    ensemble: EnsembleConfig = field(default_factory = EnsembleConfig)
    economic_validation: EconomicValidationConfig = field(default_factory = EconomicValidationConfig)
    ml_transition: MLTransitionConfig = field(default_factory = MLTransitionConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory = FeatureEngineeringConfig)
    memory: MemoryConfig = field(default_factory = MemoryConfig)
    logging: LoggingConfig = field(default_factory = LoggingConfig)
    
    # Global settings
    symbol: str = "ETHUSDT"
    exchange: str = "BINANCE"
    timeframe: str = "1m"
    data_dir: str = "data_cache"
    force_rerun: bool = False
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    cache_dir: str = "cache"
    
    # Validation settings
    enable_cross_validation: bool = True
    validation_split: float = 0.2
    enable_early_stopping: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Step03Config':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update component configurations
        if 'bayesian_optimization' in config_dict:
            config.bayesian_optimization = BayesianOptimizationConfig(**config_dict['bayesian_optimization'])
        if 'hmm' in config_dict:
            config.hmm = HMMConfig(**config_dict['hmm'])
        if 'ensemble' in config_dict:
            config.ensemble = EnsembleConfig(**config_dict['ensemble'])
        if 'economic_validation' in config_dict:
            config.economic_validation = EconomicValidationConfig(**config_dict['economic_validation'])
        if 'ml_transition' in config_dict:
            config.ml_transition = MLTransitionConfig(**config_dict['ml_transition'])
        if 'feature_engineering' in config_dict:
            config.feature_engineering = FeatureEngineeringConfig(**config_dict['feature_engineering'])
        if 'memory' in config_dict:
            config.memory = MemoryConfig(**config_dict['memory'])
        if 'logging' in config_dict:
            config.logging = LoggingConfig(**config_dict['logging'])
        
        # Update global settings
        for key, value in config_dict.items():
            if hasattr(config, key) and not key.startswith('_'):
                setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'bayesian_optimization': self.bayesian_optimization.__dict__,
            'hmm': self.hmm.__dict__,
            'ensemble': self.ensemble.__dict__,
            'economic_validation': self.economic_validation.__dict__,
            'ml_transition': self.ml_transition.__dict__,
            'feature_engineering': self.feature_engineering.__dict__,
            'memory': self.memory.__dict__,
            'logging': self.logging.__dict__,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframe': self.timeframe,
            'data_dir': self.data_dir,
            'force_rerun': self.force_rerun,
            'enable_parallel_processing': self.enable_parallel_processing,
            'max_workers': self.max_workers,
            'enable_caching': self.enable_caching,
            'cache_dir': self.cache_dir,
            'enable_cross_validation': self.enable_cross_validation,
            'validation_split': self.validation_split,
            'enable_early_stopping': self.enable_early_stopping,
        }
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents = True, exist_ok = True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent = 2)
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'Step03Config':
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif '.' in key:
                # Handle nested updates like 'hmm.n_components_range'
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)


# Default configuration instance
DEFAULT_CONFIG = Step03Config()

# Configuration presets
PRESETS = {
    'fast': Step03Config(
        bayesian_optimization = BayesianOptimizationConfig(n_trials = 20, timeout_minutes = 5),
        hmm = HMMConfig(n_components_range=[2, 8], max_samples = 2000),
        ensemble = EnsembleConfig(kmeans_n_clusters_range=[5, 15]),
        ml_transition = MLTransitionConfig(initial_features = 10, max_features = 50),
        memory = MemoryConfig(chunk_size = 5000)
    ),
    
    'thorough': Step03Config(
        bayesian_optimization = BayesianOptimizationConfig(n_trials = 200, timeout_minutes = 60),
        hmm = HMMConfig(n_components_range=[2, 40], max_samples = 10000),
        ensemble = EnsembleConfig(kmeans_n_clusters_range=[10, 50]),
        ml_transition = MLTransitionConfig(initial_features = 30, max_features = 150),
        memory = MemoryConfig(chunk_size = 20000)
    ),
    
    'production': Step03Config(
        bayesian_optimization = BayesianOptimizationConfig(n_trials = 100, timeout_minutes = 30),
        hmm = HMMConfig(n_components_range=[3, 20], max_samples = 5000),
        ensemble = EnsembleConfig(kmeans_n_clusters_range=[15, 25]),
        ml_transition = MLTransitionConfig(initial_features = 20, max_features = 100),
        memory = MemoryConfig(chunk_size = 10000, max_memory_usage_gb = 16.0),
        enable_caching = True,
        enable_parallel_processing = True
    )
}


def get_config(preset: Optional[str] = None, **kwargs) -> Step03Config:
    """Get configuration with optional preset and overrides."""
    if preset and preset in PRESETS:
        config = PRESETS[preset]
    else:
        config = DEFAULT_CONFIG
    
    if kwargs:
        config.update_from_dict(kwargs)
    
    return config