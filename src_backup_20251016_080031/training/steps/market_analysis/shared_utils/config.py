"""
Shared configuration validation and management utilities.

This module provides common configuration handling functionality that eliminates
redundancy between NAS and TAS components.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from src.utils.tprint import tprint, tprint_debug, tprint_success, tprint_warning, tprint_error


@dataclass
class BaseConfig:
    """Base configuration class with common validation methods."""
    symbol: str = "ETHUSDT"
    timeframe: str = "4h"  # Updated to 4h for regime detection
    n_regimes: int = 8
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        min_regimes = getattr(self, 'regime_search_min', 5)
        max_regimes = getattr(self, 'regime_search_max', 15)
        validate_regime_count(self.n_regimes, min_regimes, max_regimes)
        validate_algorithm_type(self.timeframe, ['1m', '5m', '15m', '30m', '1h', '4h', '1d'])


@dataclass
class NASConfig(BaseConfig):
    """Configuration for NAS regime detection."""
    primary_architecture: str = "hybrid"
    search_strategy: str = "evolutionary"
    population_size: int = 50
    generations: int = 100
    enable_neural_odes: bool = True
    enable_vision_transformers: bool = True
    enable_meta_learning: bool = True
    enable_economic_evaluation: bool = True
    enable_trading_viability: bool = True


@dataclass
class TASConfig(BaseConfig):
    """Configuration for TAS regime detection."""
    tree_depth: int = 6
    n_estimators: int = 1000
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: str = 'sqrt'
    enable_patchtst_enhancement: bool = True
    enable_statistical_methods: bool = True
    enable_economic_evaluation: bool = True
    enable_meta_learning: bool = True


@dataclass
class HybridConfig(BaseConfig):
    """Configuration for hybrid NAS-TAS regime detection."""
    combination_strategy: str = "ensemble"
    enable_nas: bool = True
    enable_tas: bool = True
    enable_consensus_analysis: bool = True
    enable_economic_evaluation: bool = True
    enable_trading_viability: bool = True
    
    # Consensus and disagreement thresholds
    consensus_threshold: float = 0.6
    disagreement_tolerance: float = 0.3
    
    # Economic and trading weights
    economic_weight: float = 0.4
    trading_weight: float = 0.3
    stability_weight: float = 0.3
    
    # System-specific configs
    nas_config: NASConfig = field(default_factory=NASConfig)
    tas_config: TASConfig = field(default_factory=TASConfig)


class ConfigValidator:
    """Configuration validator for NAS-TAS components."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the configuration validator.
        
        Args:
            verbose: Whether to enable verbose logging
        """
        self.verbose = verbose
    
    def validate_config(self, config: Any) -> List[str]:
        """
        Validate a configuration object.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Check required attributes
            required_attrs = ['symbol', 'timeframe', 'n_regimes']
            for attr in required_attrs:
                if not hasattr(config, attr):
                    errors.append(f"Missing required attribute: {attr}")
            
            if errors:
                return errors
            
            # Validate individual parameters
            errors.extend(self._validate_symbol(config.symbol))
            errors.extend(self._validate_timeframe(config.timeframe))
            min_regimes = getattr(config, 'regime_search_min', None)
            max_regimes = getattr(config, 'regime_search_max', None)
            errors.extend(
                self._validate_regime_count(
                    config.n_regimes,
                    min_regimes=min_regimes,
                    max_regimes=max_regimes,
                )
            )
            
            # Validate algorithm-specific parameters
            if hasattr(config, 'algorithm_type'):
                errors.extend(self._validate_algorithm_type(config.algorithm_type))
            
            # Validate weights if present
            if hasattr(config, 'economic_weight'):
                errors.extend(self._validate_weights(config))
            
            if self.verbose and not errors:
                tprint_success("✅ [CONFIG_VALIDATOR] Configuration validation passed")
            elif self.verbose and errors:
                tprint_error(f"❌ [CONFIG_VALIDATOR] Configuration validation failed: {errors}")
            
            return errors
            
        except Exception as e:
            error_msg = f"Configuration validation error: {e}"
            errors.append(error_msg)
            if self.verbose:
                tprint_error(f"❌ [CONFIG_VALIDATOR] {error_msg}")
            return errors
    
    def _validate_symbol(self, symbol: str) -> List[str]:
        """Validate symbol parameter."""
        errors = []
        if not isinstance(symbol, str) or not symbol.strip():
            errors.append("Symbol must be a non-empty string")
        elif len(symbol) < 3:
            errors.append("Symbol must be at least 3 characters long")
        return errors
    
    def _validate_timeframe(self, timeframe: str) -> List[str]:
        """Validate timeframe parameter."""
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        errors = []
        if not isinstance(timeframe, str) or timeframe not in valid_timeframes:
            errors.append(f"Timeframe must be one of {valid_timeframes}")
        return errors
    
    def _validate_regime_count(self, n_regimes: int, *,
                               min_regimes: Optional[int] = None,
                               max_regimes: Optional[int] = None) -> List[str]:
        """Validate regime count parameter."""
        errors = []
        lower_bound = 5 if min_regimes is None else min_regimes
        upper_bound = 15 if max_regimes is None else max_regimes

        if not isinstance(n_regimes, int) or n_regimes < lower_bound or n_regimes > upper_bound:
            errors.append(
                f"Number of regimes must be an integer between {lower_bound} and {upper_bound}"
            )
        return errors
    
    def _validate_algorithm_type(self, algorithm_type: str) -> List[str]:
        """Validate algorithm type parameter."""
        valid_algorithms = ['kmeans', 'gmm', 'hierarchical', 'dbscan', 'adaptive_clustering', 'ensemble_clustering', 'nas_tas_clustering']
        errors = []
        if not isinstance(algorithm_type, str) or algorithm_type not in valid_algorithms:
            errors.append(f"Algorithm type must be one of {valid_algorithms}")
        return errors
    
    def _validate_weights(self, config: Any) -> List[str]:
        """Validate weight parameters."""
        errors = []
        weight_attrs = ['economic_weight', 'trading_weight', 'stability_weight']
        
        for attr in weight_attrs:
            if hasattr(config, attr):
                weight = getattr(config, attr)
                if not isinstance(weight, (int, float)) or weight < 0 or weight > 1:
                    errors.append(f"{attr} must be a number between 0 and 1")
        
        # Check that weights sum to approximately 1
        if all(hasattr(config, attr) for attr in weight_attrs):
            total_weight = sum(getattr(config, attr) for attr in weight_attrs)
            if abs(total_weight - 1.0) > 0.01:
                errors.append(f"Weights must sum to approximately 1.0, got {total_weight:.3f}")
        
        return errors


def validate_regime_count(n_regimes: int, min_regimes: int = 5, max_regimes: int = 15) -> bool:
    """
    Validate regime count parameter.
    
    Args:
        n_regimes: Number of regimes to validate
        min_regimes: Minimum allowed regimes
        max_regimes: Maximum allowed regimes
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If regime count is invalid
    """
    if not isinstance(n_regimes, int):
        raise ValueError(f"Number of regimes must be an integer, got {type(n_regimes)}")
    
    if n_regimes < min_regimes or n_regimes > max_regimes:
        raise ValueError(f"Number of regimes must be between {min_regimes} and {max_regimes}, got {n_regimes}")
    
    return True


def normalize_weights(weights_dict: Dict[str, float], target_sum: float = 1.0) -> Dict[str, float]:
    """
    Normalize weights to sum to target value.
    
    Args:
        weights_dict: Dictionary of weights to normalize
        target_sum: Target sum for normalized weights
        
    Returns:
        Dictionary with normalized weights
    """
    if not weights_dict:
        return weights_dict
    
    current_sum = sum(weights_dict.values())
    if current_sum == 0:
        # If all weights are zero, distribute equally
        equal_weight = target_sum / len(weights_dict)
        return {key: equal_weight for key in weights_dict.keys()}
    
    # Normalize weights
    normalization_factor = target_sum / current_sum
    normalized_weights = {
        key: weight * normalization_factor
        for key, weight in weights_dict.items()
    }
    
    return normalized_weights


def validate_algorithm_type(algorithm_name: str, valid_algorithms: List[str]) -> bool:
    """
    Validate algorithm type parameter.
    
    Args:
        algorithm_name: Algorithm name to validate
        valid_algorithms: List of valid algorithm names
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If algorithm type is invalid
    """
    if not isinstance(algorithm_name, str):
        raise ValueError(f"Algorithm name must be a string, got {type(algorithm_name)}")
    
    if algorithm_name not in valid_algorithms:
        raise ValueError(f"Algorithm must be one of {valid_algorithms}, got {algorithm_name}")
    
    return True


def create_default_config(
    config_type: str = "hybrid",
    symbol: str = "ETHUSDT",
    timeframe: str = "4h",  # Updated to 4h for regime detection
    n_regimes: int = 8,
    **kwargs
) -> Union[NASConfig, TASConfig, HybridConfig]:
    """
    Create a default configuration for NAS-TAS components.
    
    Args:
        config_type: Type of configuration ('nas', 'tas', 'hybrid')
        symbol: Trading symbol
        timeframe: Timeframe for analysis
        n_regimes: Number of regimes
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration object
    """
    base_params = {
        'symbol': symbol,
        'timeframe': timeframe,
        'n_regimes': n_regimes,
    }
    
    if config_type.lower() == 'nas':
        # For NAS config, include all parameters
        return NASConfig(**base_params, **kwargs)
    elif config_type.lower() == 'tas':
        # For TAS config, include all parameters
        return TASConfig(**base_params, **kwargs)
    elif config_type.lower() == 'hybrid':
        # For Hybrid config, only use base parameters
        # NAS-specific parameters should be handled in nas_config
        return HybridConfig(**base_params)
    else:
        raise ValueError(f"Unknown config type: {config_type}. Must be 'nas', 'tas', or 'hybrid'")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override the value
            merged[key] = value
    
    return merged


def validate_timeframe_consistency(configs: List[Any]) -> bool:
    """
    Validate that all configurations have consistent timeframes.
    
    Args:
        configs: List of configuration objects
        
    Returns:
        True if all timeframes are consistent, False otherwise
    """
    if not configs:
        return True
    
    # Get timeframes from all configs
    timeframes = []
    for config in configs:
        if hasattr(config, 'timeframe'):
            timeframes.append(config.timeframe)
        elif hasattr(config, 'nas_config') and hasattr(config.nas_config, 'timeframe'):
            timeframes.append(config.nas_config.timeframe)
        elif hasattr(config, 'tas_config') and hasattr(config.tas_config, 'timeframe'):
            timeframes.append(config.tas_config.timeframe)
    
    # Check if all timeframes are the same
    return len(set(timeframes)) <= 1


def create_adaptive_config(
    data_size: int,
    config_type: str = "hybrid",
    **base_params
) -> Union[NASConfig, TASConfig, HybridConfig]:
    """
    Create an adaptive configuration based on data size.
    
    Args:
        data_size: Size of the dataset
        config_type: Type of configuration to create
        **base_params: Additional base parameters
        
    Returns:
        Adaptive configuration object
    """
    # Determine configuration based on data size
    if data_size < 1000:
        # Small dataset configuration
        adaptive_params = {
            'n_regimes': 2,
            'population_size': 20,
            'generations': 50,
            'tree_depth': 4,
            'n_estimators': 100,
            **base_params
        }
    elif data_size < 5000:
        # Medium dataset configuration
        adaptive_params = {
            'n_regimes': 8,
            'population_size': 50,
            'generations': 100,
            'tree_depth': 6,
            'n_estimators': 500,
            **base_params
        }
    else:
        # Large dataset configuration
        adaptive_params = {
            'n_regimes': 10,
            'population_size': 100,
            'generations': 200,
            'tree_depth': 8,
            'n_estimators': 1000,
            **base_params
        }
    
    return create_default_config(config_type, **adaptive_params)