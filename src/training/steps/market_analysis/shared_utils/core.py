"""
Shared utilities for clustering components.

This module provides essential utilities that were previously in the deleted shared_utils folder.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


@dataclass
class FeatureConfig:
    """Configuration for feature preparation."""
    n_features: int = 50
    use_pca: bool = True
    pca_components: int = 20
    use_umap: bool = False
    umap_components: int = 10
    scaler_type: str = 'robust'


@dataclass
class FeaturePreparationResult:
    """Result from feature preparation."""
    features: np.ndarray
    feature_names: List[str]
    scaler: Any
    pca: Optional[Any] = None
    umap: Optional[Any] = None
    feature_scores: Dict[str, float] = None


def validate_regime_count(regime_count: int) -> bool:
    """Validate regime count."""
    return isinstance(regime_count, int) and 2 <= regime_count <= 20


def normalize_weights(weights: List[float]) -> List[float]:
    """Normalize weights to sum to 1."""
    if not weights:
        return []
    total = sum(weights)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def validate_algorithm_type(algorithm: str) -> bool:
    """Validate clustering algorithm type."""
    valid_algorithms = ['kmeans', 'gmm', 'hdbscan', 'dbscan', 'agglomerative']
    return algorithm.lower() in valid_algorithms


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'n_regimes': 3,
        'algorithm': 'kmeans',
        'features': FeatureConfig(),
        'validation': True,
        'optimization': True
    }


@dataclass
class BaseConfig:
    """Base configuration class."""
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the configuration."""
        validator = ConfigValidator(verbose=True)
        if not validator.validate_config(self):
            raise ValueError("Configuration validation failed")


class ConfigValidator:
    """Configuration validator."""
    
    def __init__(self, verbose: bool = False):
        """Initialize validator."""
        self.verbose = verbose
        self.logger = get_logger('ConfigValidator')
    
    def validate_config(self, config: Any) -> bool:
        """Validate configuration object."""
        try:
            if hasattr(config, '__dict__'):
                return self._validate_dict(config.__dict__)
            elif isinstance(config, dict):
                return self._validate_dict(config)
            else:
                self.logger.error("Invalid config type: must be dict or object with __dict__")
                return False
        except Exception as e:
            self.logger.error(f"Config validation failed: {e}")
            return False
    
    def _validate_dict(self, config_dict: Dict[str, Any]) -> bool:
        """Validate configuration dictionary."""
        try:
            # Check required fields
            required_fields = ['n_regimes', 'algorithm_type']
            for field in required_fields:
                if field not in config_dict:
                    if self.verbose:
                        self.logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate n_regimes
            n_regimes = config_dict.get('n_regimes')
            if not isinstance(n_regimes, int) or not (2 <= n_regimes <= 25):
                if self.verbose:
                    self.logger.error(f"Invalid n_regimes: {n_regimes} (must be int between 2-25)")
                return False
            
            # Validate algorithm_type
            algorithm_type = config_dict.get('algorithm_type')
            valid_algorithms = ['gaussian_mixture', 'kmeans', 'agglomerative', 'adaptive_clustering']
            if algorithm_type not in valid_algorithms:
                if self.verbose:
                    self.logger.error(f"Invalid algorithm_type: {algorithm_type} (must be one of {valid_algorithms})")
                return False
            
            # Validate weights if present
            weight_fields = ['economic_weight', 'volatility_regime_weight', 'volume_regime_weight', 'structural_trend_weight']
            if all(field in config_dict for field in weight_fields):
                total_weight = sum(config_dict[field] for field in weight_fields)
                if not (0.9 <= total_weight <= 1.1):  # Allow some tolerance
                    if self.verbose:
                        self.logger.warning(f"Weights sum to {total_weight:.3f}, should be close to 1.0")
            
            if self.verbose:
                self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Validation error: {e}")
            return False


def log_execution(*args, **kwargs):
    """
    Log execution function that can be used as both a regular function and a decorator.
    
    When used as decorator: @log_execution('func_name', 'context', verbose=True)
    When used as function: log_execution('func_name', context='context', verbose=True)
    """
    logger = get_logger('execution')
    
    # Check if this is being used as a decorator (called with arguments)
    if len(args) >= 1 and not callable(args[0]):
        # Used as decorator with arguments: @log_execution('name', 'context', verbose=True)
        func_name = args[0]
        context = args[1] if len(args) > 1 else ""
        verbose = kwargs.get('verbose', False)
        
        def decorator(func):
            def wrapper(*wrapper_args, **wrapper_kwargs):
                logger.info(f"Executing {func_name} in context {context} (verbose: {verbose})")
                return func(*wrapper_args, **wrapper_kwargs)
            return wrapper
        return decorator
    
    elif len(args) == 1 and callable(args[0]):
        # Used as decorator without arguments: @log_execution
        func = args[0]
        
        def wrapper(*wrapper_args, **wrapper_kwargs):
            logger.info(f"Executing {func.__name__}")
            return func(*wrapper_args, **wrapper_kwargs)
        return wrapper
    
    else:
        # Used as regular function
        func_name = args[0] if args else "unknown"
        context = kwargs.get('context', "")
        verbose = kwargs.get('verbose', False)
        logger.info(f"Executing {func_name} in context {context} (verbose: {verbose})")


def log_execution_decorator(func_name: str, context: str = "", verbose: bool = False):
    """
    Decorator factory for log_execution.
    
    Args:
        func_name: Name of the function being decorated
        context: Context string for logging
        verbose: Whether to use verbose logging
    """
    def decorator(func):
        logger = get_logger('execution')
        
        def wrapper(*args, **kwargs):
            logger.info(f"Executing {func_name} in context {context} (verbose: {verbose})")
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def log_execution_with_context(func_name: str, context: str = "", **kwargs):
    """Log execution with context."""
    logger = get_logger('execution')
    logger.info(f"Executing {func_name} in context {context} with {kwargs}")


def log_performance(metric: str, value: float, **kwargs):
    """Log performance metric."""
    logger = get_logger('performance')
    logger.info(f"{metric}: {value}")


class LoggingContext:
    """Logging context manager."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(name)
    
    def __enter__(self):
        self.logger.info(f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(f"Error in {self.name}: {exc_val}")
        else:
            self.logger.info(f"Completed {self.name}")


def calculate_consensus_metrics(data: np.ndarray) -> Dict[str, float]:
    """Calculate consensus metrics."""
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data)
    }


def calculate_disagreement_metrics(data: np.ndarray) -> Dict[str, float]:
    """Calculate disagreement metrics."""
    return {
        'variance': np.var(data),
        'range': np.max(data) - np.min(data),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25)
    }


def calculate_economic_scores(data: np.ndarray) -> Dict[str, float]:
    """Calculate economic scores."""
    return {
        'return': np.mean(data),
        'volatility': np.std(data),
        'sharpe': np.mean(data) / np.std(data) if np.std(data) > 0 else 0
    }


def calculate_trading_scores(data: np.ndarray) -> Dict[str, float]:
    """Calculate trading scores."""
    return {
        'frequency': len(data),
        'consistency': 1.0 - np.std(data) / np.mean(data) if np.mean(data) > 0 else 0
    }


def calculate_stability_scores(data: np.ndarray) -> Dict[str, float]:
    """Calculate stability scores."""
    return {
        'stability': 1.0 - np.std(data) / np.mean(data) if np.mean(data) > 0 else 0,
        'trend': np.polyfit(range(len(data)), data, 1)[0] if len(data) > 1 else 0
    }


class MetricsCalculator:
    """Metrics calculator."""
    
    def __init__(self):
        self.logger = get_logger('metrics')
    
    def calculate_all(self, data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate all metrics."""
        return {
            'consensus': calculate_consensus_metrics(data),
            'disagreement': calculate_disagreement_metrics(data),
            'economic': calculate_economic_scores(data),
            'trading': calculate_trading_scores(data),
            'stability': calculate_stability_scores(data)
        }


def create_regime_characteristics(data: np.ndarray) -> Dict[str, Any]:
    """Create regime characteristics."""
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'count': len(data),
        'range': np.max(data) - np.min(data)
    }


def generate_cluster_characteristics(data: np.ndarray) -> Dict[str, Any]:
    """Generate cluster characteristics."""
    return create_regime_characteristics(data)


class CharacteristicsGenerator:
    """Characteristics generator."""
    
    def __init__(self):
        self.logger = get_logger('characteristics')
    
    def generate(self, data: np.ndarray) -> Dict[str, Any]:
        """Generate characteristics."""
        return generate_cluster_characteristics(data)


def prepare_market_features(
    market_data: pd.DataFrame,
    config: FeatureConfig
) -> FeaturePreparationResult:
    """
    Prepare market features for clustering.
    
    Args:
        market_data: Market data DataFrame
        config: Feature configuration
        
    Returns:
        FeaturePreparationResult with prepared features
    """
    logger = get_logger('prepare_market_features')
    
    try:
        # Basic feature preparation
        features = market_data.select_dtypes(include=[np.number]).values
        
        # Simple feature names
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        # Apply scaling
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply PCA if requested
        pca = None
        if config.use_pca and config.pca_components < features_scaled.shape[1]:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=config.pca_components)
            features_scaled = pca.fit_transform(features_scaled)
            feature_names = [f"pca_{i}" for i in range(config.pca_components)]
        
        # Apply UMAP if requested
        umap = None
        if config.use_umap:
            try:
                import umap
                umap = umap.UMAP(n_components=config.umap_components)
                features_scaled = umap.fit_transform(features_scaled)
                feature_names = [f"umap_{i}" for i in range(config.umap_components)]
            except ImportError:
                logger.warning("UMAP not available, skipping UMAP transformation")
        
        return FeaturePreparationResult(
            features=features_scaled,
            feature_names=feature_names,
            scaler=scaler,
            pca=pca,
            umap=umap,
            feature_scores={name: 1.0 for name in feature_names}
        )
        
    except Exception as e:
        logger.error(f"Feature preparation failed: {e}")
        raise


def calculate_consensus_metrics(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate consensus metrics for clustering results."""
    try:
        n_clusters = len(np.unique(cluster_assignments))
        n_samples = len(cluster_assignments)
        
        if n_clusters <= 1:
            return {
                'n_clusters': n_clusters,
                'n_samples': n_samples,
                'consensus_score': 0.0,
                'stability_score': 0.0
            }
        
        # Calculate consensus score based on cluster size distribution
        cluster_sizes = [np.sum(cluster_assignments == i) for i in range(n_clusters)]
        size_std = np.std(cluster_sizes)
        size_mean = np.mean(cluster_sizes)
        consensus_score = 1.0 - (size_std / size_mean) if size_mean > 0 else 0.0
        consensus_score = max(0.0, min(1.0, consensus_score))
        
        # Calculate stability score based on cluster separation
        if hasattr(market_data, 'values'):
            data_values = market_data.values
        else:
            data_values = market_data
            
        # Simple stability measure based on within-cluster variance
        within_cluster_var = 0.0
        for i in range(n_clusters):
            cluster_mask = cluster_assignments == i
            if np.sum(cluster_mask) > 1:
                cluster_data = data_values[cluster_mask]
                within_cluster_var += np.var(cluster_data)
        
        total_var = np.var(data_values)
        stability_score = 1.0 - (within_cluster_var / total_var) if total_var > 0 else 0.0
        stability_score = max(0.0, min(1.0, stability_score))
        
        return {
            'n_clusters': n_clusters,
            'n_samples': n_samples,
            'consensus_score': consensus_score,
            'stability_score': stability_score
        }
    except Exception as e:
        logger = get_logger('consensus_metrics')
        logger.error(f"Consensus metrics calculation failed: {e}")
        return {'n_clusters': 0, 'n_samples': 0, 'consensus_score': 0.0, 'stability_score': 0.0}


def calculate_disagreement_metrics(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate disagreement metrics for clustering results."""
    try:
        n_clusters = len(np.unique(cluster_assignments))
        n_samples = len(cluster_assignments)
        
        if n_clusters <= 1:
            return {
                'disagreement_score': 0.0,
                'uncertainty_score': 0.0
            }
        
        # Calculate disagreement score based on cluster size imbalance
        cluster_sizes = [np.sum(cluster_assignments == i) for i in range(n_clusters)]
        max_size = max(cluster_sizes)
        min_size = min(cluster_sizes)
        disagreement_score = (max_size - min_size) / n_samples if n_samples > 0 else 0.0
        disagreement_score = max(0.0, min(1.0, disagreement_score))
        
        # Calculate uncertainty score based on cluster boundary ambiguity
        if hasattr(market_data, 'values'):
            data_values = market_data.values
        else:
            data_values = market_data
            
        # Simple uncertainty measure based on data variance
        data_var = np.var(data_values)
        data_mean = np.mean(data_values)
        uncertainty_score = data_var / (data_mean ** 2) if data_mean != 0 else 0.0
        uncertainty_score = max(0.0, min(1.0, uncertainty_score))
        
        return {
            'disagreement_score': disagreement_score,
            'uncertainty_score': uncertainty_score
        }
    except Exception as e:
        logger = get_logger('disagreement_metrics')
        logger.error(f"Disagreement metrics calculation failed: {e}")
        return {'disagreement_score': 0.0, 'uncertainty_score': 0.0}


def calculate_economic_scores(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate economic scores for clustering results."""
    try:
        n_clusters = len(np.unique(cluster_assignments))
        
        if n_clusters <= 1:
            return {
                'economic_score': 0.0,
                'trading_score': 0.0
            }
        
        # Calculate economic score based on cluster separation and market data
        if hasattr(market_data, 'values'):
            data_values = market_data.values
        else:
            data_values = market_data
            
        # Economic score based on cluster distinctiveness
        cluster_means = []
        for i in range(n_clusters):
            cluster_mask = cluster_assignments == i
            if np.sum(cluster_mask) > 0:
                cluster_data = data_values[cluster_mask]
                cluster_means.append(np.mean(cluster_data))
        
        if len(cluster_means) > 1:
            mean_std = np.std(cluster_means)
            mean_mean = np.mean(cluster_means)
            economic_score = mean_std / (mean_mean + 1e-8) if mean_mean != 0 else 0.0
            economic_score = max(0.0, min(1.0, economic_score))
        else:
            economic_score = 0.0
        
        # Trading score based on cluster consistency
        cluster_consistency = []
        for i in range(n_clusters):
            cluster_mask = cluster_assignments == i
            if np.sum(cluster_mask) > 1:
                cluster_data = data_values[cluster_mask]
                cluster_std = np.std(cluster_data)
                cluster_mean = np.mean(cluster_data)
                consistency = 1.0 - (cluster_std / (cluster_mean + 1e-8)) if cluster_mean != 0 else 0.0
                cluster_consistency.append(max(0.0, min(1.0, consistency)))
        
        trading_score = np.mean(cluster_consistency) if cluster_consistency else 0.0
        
        return {
            'economic_score': economic_score,
            'trading_score': trading_score
        }
    except Exception as e:
        logger = get_logger('economic_scores')
        logger.error(f"Economic scores calculation failed: {e}")
        return {'economic_score': 0.0, 'trading_score': 0.0}


def calculate_trading_scores(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate trading scores for clustering results."""
    try:
        n_clusters = len(np.unique(cluster_assignments))
        
        if n_clusters <= 1:
            return {
                'trading_score': 0.0,
                'profitability_score': 0.0
            }
        
        # Calculate trading score based on cluster frequency and consistency
        cluster_sizes = [np.sum(cluster_assignments == i) for i in range(n_clusters)]
        total_samples = len(cluster_assignments)
        
        # Trading score based on cluster frequency distribution
        cluster_frequencies = [size / total_samples for size in cluster_sizes]
        frequency_entropy = -sum(f * np.log(f + 1e-8) for f in cluster_frequencies)
        max_entropy = np.log(n_clusters)
        trading_score = frequency_entropy / max_entropy if max_entropy > 0 else 0.0
        trading_score = max(0.0, min(1.0, trading_score))
        
        # Profitability score based on cluster value distribution
        if hasattr(market_data, 'values'):
            data_values = market_data.values
        else:
            data_values = market_data
            
        cluster_values = []
        for i in range(n_clusters):
            cluster_mask = cluster_assignments == i
            if np.sum(cluster_mask) > 0:
                cluster_data = data_values[cluster_mask]
                cluster_values.append(np.mean(cluster_data))
        
        if len(cluster_values) > 1:
            value_range = max(cluster_values) - min(cluster_values)
            value_mean = np.mean(cluster_values)
            profitability_score = value_range / (value_mean + 1e-8) if value_mean != 0 else 0.0
            profitability_score = max(0.0, min(1.0, profitability_score))
        else:
            profitability_score = 0.0
        
        return {
            'trading_score': trading_score,
            'profitability_score': profitability_score
        }
    except Exception as e:
        logger = get_logger('trading_scores')
        logger.error(f"Trading scores calculation failed: {e}")
        return {'trading_score': 0.0, 'profitability_score': 0.0}


def calculate_stability_scores(
    cluster_assignments: np.ndarray,
    market_data: pd.DataFrame
) -> Dict[str, float]:
    """Calculate stability scores for clustering results."""
    try:
        n_clusters = len(np.unique(cluster_assignments))
        
        if n_clusters <= 1:
            return {
                'stability_score': 0.0,
                'consistency_score': 0.0
            }
        
        # Calculate stability score based on cluster size consistency
        cluster_sizes = [np.sum(cluster_assignments == i) for i in range(n_clusters)]
        size_cv = np.std(cluster_sizes) / (np.mean(cluster_sizes) + 1e-8)
        stability_score = 1.0 - min(1.0, size_cv)
        stability_score = max(0.0, min(1.0, stability_score))
        
        # Calculate consistency score based on within-cluster variance
        if hasattr(market_data, 'values'):
            data_values = market_data.values
        else:
            data_values = market_data
            
        within_cluster_var = 0.0
        total_var = np.var(data_values)
        
        for i in range(n_clusters):
            cluster_mask = cluster_assignments == i
            if np.sum(cluster_mask) > 1:
                cluster_data = data_values[cluster_mask]
                within_cluster_var += np.var(cluster_data)
        
        if total_var > 0:
            consistency_score = 1.0 - (within_cluster_var / total_var)
            consistency_score = max(0.0, min(1.0, consistency_score))
        else:
            consistency_score = 0.0
        
        return {
            'stability_score': stability_score,
            'consistency_score': consistency_score
        }
    except Exception as e:
        logger = get_logger('stability_scores')
        logger.error(f"Stability scores calculation failed: {e}")
        return {'stability_score': 0.0, 'consistency_score': 0.0}


class MetricsCalculator:
    """Calculator for various clustering metrics."""
    
    def __init__(self):
        self.logger = get_logger('MetricsCalculator')
    
    def calculate_all_metrics(
        self,
        cluster_assignments: np.ndarray,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate all available metrics."""
        try:
            metrics = {}
            metrics.update(calculate_consensus_metrics(cluster_assignments, market_data))
            metrics.update(calculate_disagreement_metrics(cluster_assignments, market_data))
            metrics.update(calculate_economic_scores(cluster_assignments, market_data))
            metrics.update(calculate_trading_scores(cluster_assignments, market_data))
            metrics.update(calculate_stability_scores(cluster_assignments, market_data))
            return metrics
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            return {}


class CharacteristicsGenerator:
    """Generator for cluster characteristics."""
    
    def __init__(self):
        self.logger = get_logger('CharacteristicsGenerator')
    
    def generate_characteristics(
        self,
        cluster_assignments: np.ndarray,
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate cluster characteristics."""
        try:
            n_clusters = len(np.unique(cluster_assignments))
            characteristics = {
                'n_clusters': n_clusters,
                'cluster_sizes': [np.sum(cluster_assignments == i) for i in range(n_clusters)],
                'cluster_characteristics': {}
            }
            
            for i in range(n_clusters):
                cluster_mask = cluster_assignments == i
                cluster_data = market_data[cluster_mask]
                
                characteristics['cluster_characteristics'][f'cluster_{i}'] = {
                    'size': np.sum(cluster_mask),
                    'mean_volume': cluster_data['volume'].mean() if 'volume' in cluster_data.columns else 0,
                    'mean_price': cluster_data['close'].mean() if 'close' in cluster_data.columns else 0
                }
            
            return characteristics
        except Exception as e:
            self.logger.error(f"Characteristics generation failed: {e}")
            return {'n_clusters': 0, 'cluster_sizes': [], 'cluster_characteristics': {}}