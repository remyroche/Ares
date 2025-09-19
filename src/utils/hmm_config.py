#!/usr/bin/env python3
"""
Unified HMM Configuration Module

This module consolidates all HMM-related configuration classes that were previously
scattered across multiple files. It provides a single, comprehensive configuration
system for all HMM operations.

Replaces duplicate configurations from:
- market_analysis/hmm_clustering/config.py (HMMClusteringConfig)
- utils/ml_common/hmm_regime_detection.py (HMMRegimeConfig)  
- training/steps/market_analysis/hmm_clustering_config.py (UnifiedHMMClusteringConfig)
- utils/ml_common/config/base_training_config.py (HMMTrainingConfig)
- utils/data/quality/data_qualification_config.py (HMMRegimeConfig)
- And several other duplicate config classes
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import json
from pathlib import Path


class HMMCovarianceType(Enum):
    """Supported covariance types for HMM models."""
    SPHERICAL = "spherical"
    DIAG = "diag"
    FULL = "full"
    TIED = "tied"


class HMMRegimeDetectionMethod(Enum):
    """Methods for HMM regime detection."""
    GAUSSIAN = "gaussian"
    MULTIVARIATE = "multivariate"
    ENSEMBLE = "ensemble"
    MULTI_TIMEFRAME = "multi_timeframe"
    STREAMING = "streaming"
    REGIME_AWARE = "regime_aware"


class FeatureSelectionMethod(Enum):
    """Feature selection methods."""
    MRMR = "mrmr"
    MUTUAL_INFO = "mutual_info"
    LASSO = "lasso"
    RANDOM_FOREST = "random_forest"
    UNIVARIATE = "univariate"
    PCA = "pca"
    NONE = "none"


@dataclass
class HMMModelConfig:
    """Core HMM model configuration."""
    # Basic HMM parameters
    n_components: int = 3
    covariance_type: Union[str, HMMCovarianceType] = HMMCovarianceType.FULL
    n_iter: int = 100
    tol: float = 1e-4
    random_state: int = 42
    
    # Algorithm parameters
    algorithm: str = "viterbi"
    init_params: str = "stmc"  # states, transitions, means, covariances
    params: str = "stmc"
    
    # Convergence parameters
    max_iter: int = 100
    min_covar: float = 1e-3
    startprob_prior: float = 1.0
    transmat_prior: float = 1.0
    
    def to_hmmlearn_params(self) -> Dict[str, Any]:
        """Convert to hmmlearn-compatible parameters."""
        covar_type = self.covariance_type
        if isinstance(covar_type, HMMCovarianceType):
            covar_type = covar_type.value
            
        return {
            'n_components': self.n_components,
            'covariance_type': covar_type,
            'n_iter': self.n_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            'algorithm': self.algorithm,
            'init_params': self.init_params,
            'params': self.params,
            'min_covar': self.min_covar,
            'startprob_prior': self.startprob_prior,
            'transmat_prior': self.transmat_prior
        }


@dataclass
class HMMFeatureConfig:
    """Feature engineering configuration for HMM."""
    # Technical indicators
    lookback_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    technical_indicators: List[str] = field(default_factory=lambda: [
        "rsi", "macd", "bollinger_bands", "atr", "stochastic"
    ])
    
    # Feature selection
    feature_selection_method: Union[str, FeatureSelectionMethod] = FeatureSelectionMethod.MRMR
    max_features: int = 50
    min_feature_importance: float = 0.01
    
    # Data processing
    min_data_points: int = 1000
    max_missing_ratio: float = 0.1
    normalize_features: bool = True
    remove_outliers: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 3.0


@dataclass
class HMMOptimizationConfig:
    """Optimization configuration for HMM."""
    # Bayesian optimization
    use_bayesian_optimization: bool = True
    n_trials: int = 50
    timeout_minutes: int = 15
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    purged_cv: bool = True
    
    # Hardware optimization
    use_gpu: bool = True
    use_memory_optimization: bool = True
    use_cpu_optimization: bool = True
    parallel_jobs: int = -1
    
    # Ensemble parameters
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'hmm': 0.4, 'kmeans': 0.3, 'dbscan': 0.3
    })


@dataclass
class HMMValidationConfig:
    """Validation configuration for HMM."""
    # Regime analysis
    min_regime_duration: int = 10
    regime_stability_threshold: float = 0.7
    
    # Performance thresholds
    min_silhouette_score: float = 0.3
    min_regime_separation: float = 0.5
    max_regime_overlap: float = 0.3
    
    # Economic validation
    validate_economic_significance: bool = True
    min_sharpe_improvement: float = 0.1
    min_regime_return_diff: float = 0.02


@dataclass
class HMMTrainingConfig:
    """Training configuration for HMM."""
    # Training parameters
    initial_features: int = 20
    feature_increment: int = 10
    max_features_total: int = 100
    min_improvement: float = 0.001
    patience: int = 3
    
    # Data splitting
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    
    # Regime-specific training
    regime_specific_models: bool = True
    min_samples_per_regime: int = 100
    
    # Model persistence
    save_models: bool = True
    save_intermediate: bool = False
    model_versioning: bool = True


@dataclass
class UnifiedHMMConfig:
    """
    Unified HMM configuration that consolidates all HMM-related settings.
    
    This replaces multiple configuration classes:
    - HMMClusteringConfig
    - HMMRegimeConfig  
    - UnifiedHMMClusteringConfig
    - HMMTrainingConfig
    - And others
    """
    # Core components
    model: HMMModelConfig = field(default_factory=HMMModelConfig)
    features: HMMFeatureConfig = field(default_factory=HMMFeatureConfig)
    optimization: HMMOptimizationConfig = field(default_factory=HMMOptimizationConfig)
    validation: HMMValidationConfig = field(default_factory=HMMValidationConfig)
    training: HMMTrainingConfig = field(default_factory=HMMTrainingConfig)
    
    # Detection method
    detection_method: Union[str, HMMRegimeDetectionMethod] = HMMRegimeDetectionMethod.GAUSSIAN
    
    # General settings
    symbol: str = "ETHUSDT"
    exchange: str = "BINANCE"
    timeframe: str = "1h"
    
    # Output settings
    output_dir: str = "hmm_results"
    save_results: bool = True
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def _convert_value(value):
            if isinstance(value, Enum):
                return value.value
            elif hasattr(value, '__dict__'):
                return {k: _convert_value(v) for k, v in value.__dict__.items()}
            elif isinstance(value, list):
                return [_convert_value(v) for v in value]
            elif isinstance(value, dict):
                return {k: _convert_value(v) for k, v in value.items()}
            else:
                return value
        
        return _convert_value(self)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'UnifiedHMMConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert back to proper types
        config = cls()
        config._update_from_dict(data)
        return config
    
    def _update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if hasattr(attr, '__dict__'):
                    # Handle nested dataclass
                    for nested_key, nested_value in value.items():
                        if hasattr(attr, nested_key):
                            setattr(attr, nested_key, nested_value)
                else:
                    setattr(self, key, value)


# Factory functions for common configurations
def create_default_hmm_config() -> UnifiedHMMConfig:
    """Create default HMM configuration."""
    return UnifiedHMMConfig()


def create_fast_hmm_config() -> UnifiedHMMConfig:
    """Create HMM configuration optimized for speed."""
    config = UnifiedHMMConfig()
    config.model.n_iter = 50
    config.model.n_components = 2
    config.optimization.n_trials = 20
    config.optimization.timeout_minutes = 5
    config.features.max_features = 20
    config.features.lookback_windows = [10, 20]
    return config


def create_comprehensive_hmm_config() -> UnifiedHMMConfig:
    """Create comprehensive HMM configuration for thorough analysis."""
    config = UnifiedHMMConfig()
    config.model.n_iter = 200
    config.model.n_components = 4
    config.optimization.n_trials = 100
    config.optimization.timeout_minutes = 30
    config.features.max_features = 100
    config.features.lookback_windows = [5, 10, 20, 50, 100]
    config.features.technical_indicators.extend([
        "adx", "cci", "williams_r", "momentum", "roc"
    ])
    return config


def create_regime_specific_hmm_config() -> UnifiedHMMConfig:
    """Create configuration for regime-specific HMM training."""
    config = UnifiedHMMConfig()
    config.training.regime_specific_models = True
    config.training.min_samples_per_regime = 200
    config.validation.min_regime_duration = 20
    config.validation.regime_stability_threshold = 0.8
    config.detection_method = HMMRegimeDetectionMethod.REGIME_AWARE
    return config


# Backward compatibility aliases
HMMClusteringConfig = UnifiedHMMConfig
HMMRegimeConfig = UnifiedHMMConfig
UnifiedHMMClusteringConfig = UnifiedHMMConfig

# Legacy support functions
def get_hmm_clustering_config(**kwargs) -> UnifiedHMMConfig:
    """Legacy function for backward compatibility."""
    config = create_default_hmm_config()
    
    # Update with provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.features, key):
            setattr(config.features, key, value)
        elif hasattr(config.optimization, key):
            setattr(config.optimization, key, value)
    
    return config