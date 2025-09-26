#!/usr/bin/env python3
"""
Neural Architecture Search (NAS) Configuration Module

This module provides comprehensive configuration management for Neural Architecture Search,
clustering algorithms, and related ML optimization pipelines. It integrates with the existing
utility modules for enhanced functionality.

Features:
- NAS clustering configuration with validation
- Architecture type enumeration with extensible design
- Integration with M1 hardware optimizations
- Serialization and validation utilities
- ML pipeline integration
- Comprehensive logging and monitoring
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

# Optional dependencies with fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Import utility modules with fallbacks
try:
    from src.utils.common_operations import (
        safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
        validate_finite, validate_positive, validate_range, safe_divide,
        get_current_datetime, format_datetime, safe_deepcopy, safe_copy
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError:
    COMMON_OPERATIONS_AVAILABLE = False
    # Fallback implementations
    def safe_json_dump(data, filepath, **kwargs):
        with open(filepath, 'w') as f:
            json.dump(data, f, **kwargs)
        return True
    def safe_json_load(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    def safe_file_exists(filepath):
        return Path(filepath).exists()
    def ensure_directory(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    def get_current_datetime():
        from datetime import datetime
        return datetime.now()
    def format_datetime(dt, format_str="%Y-%m-%d %H:%M:%S"):
        return dt.strftime(format_str)

try:
    from src.utils.common_utilities import (
        validate_dataframe_columns, create_summary_statistics, 
        get_dataframe_info, calculate_data_quality_metrics
    )
    COMMON_UTILITIES_AVAILABLE = True
except ImportError:
    COMMON_UTILITIES_AVAILABLE = False
    def calculate_data_quality_metrics(df):
        return {'total_rows': len(df) if hasattr(df, '__len__') else 0}

try:
    from src.utils.math_validation import (
        validate_numeric_array, safe_correlation, safe_covariance,
        validate_correlation_matrix, safe_matrix_inverse, MathValidation
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False
    class MathValidation:
        """Fallback math validation class when math_validation module is not available."""
        def __init__(self):
            """Initialize the fallback math validation class."""
            self.validation_enabled = True
            self.strict_mode = False
            self.allow_nan_default = False
        
        def validate_numeric_array(self, data, allow_nan=False):
            """Basic numeric array validation."""
            try:
                if data is None:
                    return False, "Data is None"
                if hasattr(data, '__len__') and len(data) == 0:
                    return False, "Data is empty"
                return True, "Valid"
            except Exception as e:
                return False, f"Validation error: {e}"
        
        def safe_correlation(self, x, y):
            """Safe correlation calculation."""
            try:
                if len(x) != len(y) or len(x) < 2:
                    return 0.0
                # Simple correlation calculation
                mean_x = sum(x) / len(x)
                mean_y = sum(y) / len(y)
                numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
                denominator = (sum((x[i] - mean_x) ** 2 for i in range(len(x))) * 
                             sum((y[i] - mean_y) ** 2 for i in range(len(y)))) ** 0.5
                return numerator / denominator if denominator != 0 else 0.0
            except Exception:
                return 0.0
        
        def safe_covariance(self, x, y):
            """Safe covariance calculation."""
            try:
                if len(x) != len(y) or len(x) < 2:
                    return 0.0
                mean_x = sum(x) / len(x)
                mean_y = sum(y) / len(y)
                return sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / (len(x) - 1)
            except Exception:
                return 0.0
        
        def validate_correlation_matrix(self, matrix):
            """Basic correlation matrix validation."""
            try:
                if matrix is None:
                    return False, "Matrix is None"
                if not hasattr(matrix, 'shape'):
                    return False, "Matrix has no shape attribute"
                if len(matrix.shape) != 2:
                    return False, "Matrix is not 2D"
                if matrix.shape[0] != matrix.shape[1]:
                    return False, "Matrix is not square"
                return True, "Valid"
            except Exception as e:
                return False, f"Validation error: {e}"
        
        def safe_matrix_inverse(self, matrix):
            """Safe matrix inverse calculation."""
            try:
                # Simple 2x2 matrix inverse for fallback
                if hasattr(matrix, 'shape') and matrix.shape == (2, 2):
                    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
                    det = a * d - b * c
                    if abs(det) < 1e-10:
                        return None
                    return [[d/det, -b/det], [-c/det, a/det]]
                return None
            except Exception:
                return None

try:
    from src.utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
    SERIALIZATION_AVAILABLE = True
except ImportError:
    SERIALIZATION_AVAILABLE = False
    class UniversalSerializer:
        def save(self, data, filepath, format='auto'):
            if filepath.endswith('.json'):
                return safe_json_dump(data, filepath)
            elif filepath.endswith('.pkl'):
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                return True
            return False

try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs):
        print(*args, **kwargs)
    tprint_info = tprint_warning = tprint_error = tprint_success = tprint

try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, is_m1_available, is_mps_available
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    HARDWARE_UTILS_AVAILABLE = True
except ImportError:
    HARDWARE_UTILS_AVAILABLE = False
    def get_m1_gpu_manager():
        return None
    def is_m1_available():
        return False
    def is_mps_available():
        return False
    def get_m1_memory_optimizer(memory_limit_gb=None):
        return None
    def get_m1_cpu_optimizer():
        return None

# Setup logging
logger = logging.getLogger(__name__)


class NASArchitectureType(Enum):
    """Enumeration of Neural Architecture Search architecture types."""
    
    # Tree-based architectures
    DECISION_TREE = auto()
    RANDOM_FOREST = auto()
    GRADIENT_BOOSTING = auto()
    XGBOOST = auto()
    LIGHTGBM = auto()
    CATBOOST = auto()
    
    # Neural network architectures
    FEEDFORWARD_NN = auto()
    CONVOLUTIONAL_NN = auto()
    RECURRENT_NN = auto()
    LSTM_NN = auto()
    GRU_NN = auto()
    TRANSFORMER = auto()
    
    # Ensemble architectures
    VOTING_ENSEMBLE = auto()
    STACKING_ENSEMBLE = auto()
    BAGGING_ENSEMBLE = auto()
    ADABOOST_ENSEMBLE = auto()
    
    # Specialized trading architectures
    REGIME_AWARE_TREE = auto()
    TRADING_ENSEMBLE = auto()
    SIGNAL_FUSION_NN = auto()
    MULTI_TIMEFRAME_TREE = auto()
    
    # Custom architectures
    CUSTOM_TREE = auto()
    CUSTOM_NN = auto()
    HYBRID_ARCHITECTURE = auto()
    
    @classmethod
    def get_tree_architectures(cls) -> List['NASArchitectureType']:
        """Get all tree-based architecture types."""
        return [
            cls.DECISION_TREE, cls.RANDOM_FOREST, cls.GRADIENT_BOOSTING,
            cls.XGBOOST, cls.LIGHTGBM, cls.CATBOOST, cls.REGIME_AWARE_TREE,
            cls.MULTI_TIMEFRAME_TREE, cls.CUSTOM_TREE
        ]
    
    @classmethod
    def get_neural_architectures(cls) -> List['NASArchitectureType']:
        """Get all neural network architecture types."""
        return [
            cls.FEEDFORWARD_NN, cls.CONVOLUTIONAL_NN, cls.RECURRENT_NN,
            cls.LSTM_NN, cls.GRU_NN, cls.TRANSFORMER, cls.SIGNAL_FUSION_NN,
            cls.CUSTOM_NN
        ]
    
    @classmethod
    def get_ensemble_architectures(cls) -> List['NASArchitectureType']:
        """Get all ensemble architecture types."""
        return [
            cls.VOTING_ENSEMBLE, cls.STACKING_ENSEMBLE, cls.BAGGING_ENSEMBLE,
            cls.ADABOOST_ENSEMBLE, cls.TRADING_ENSEMBLE, cls.HYBRID_ARCHITECTURE
        ]
    
    @classmethod
    def get_trading_architectures(cls) -> List['NASArchitectureType']:
        """Get all trading-specific architecture types."""
        return [
            cls.REGIME_AWARE_TREE, cls.TRADING_ENSEMBLE, cls.SIGNAL_FUSION_NN,
            cls.MULTI_TIMEFRAME_TREE, cls.HYBRID_ARCHITECTURE
        ]
    
    def is_tree_based(self) -> bool:
        """Check if architecture is tree-based."""
        return self in self.get_tree_architectures()
    
    def is_neural_network(self) -> bool:
        """Check if architecture is neural network-based."""
        return self in self.get_neural_architectures()
    
    def is_ensemble(self) -> bool:
        """Check if architecture is ensemble-based."""
        return self in self.get_ensemble_architectures()
    
    def is_trading_specific(self) -> bool:
        """Check if architecture is trading-specific."""
        return self in self.get_trading_architectures()
    
    def get_complexity_factor(self) -> float:
        """Get complexity factor for this architecture type."""
        complexity_map = {
            # Tree-based (lower complexity)
            self.DECISION_TREE: 1.0,
            self.RANDOM_FOREST: 2.5,
            self.GRADIENT_BOOSTING: 3.0,
            self.XGBOOST: 3.5,
            self.LIGHTGBM: 3.2,
            self.CATBOOST: 3.8,
            
            # Neural networks (higher complexity)
            self.FEEDFORWARD_NN: 4.0,
            self.CONVOLUTIONAL_NN: 6.0,
            self.RECURRENT_NN: 5.0,
            self.LSTM_NN: 6.5,
            self.GRU_NN: 6.0,
            self.TRANSFORMER: 8.0,
            
            # Ensembles (variable complexity)
            self.VOTING_ENSEMBLE: 2.0,
            self.STACKING_ENSEMBLE: 5.0,
            self.BAGGING_ENSEMBLE: 3.0,
            self.ADABOOST_ENSEMBLE: 4.0,
            
            # Trading-specific (medium-high complexity)
            self.REGIME_AWARE_TREE: 4.5,
            self.TRADING_ENSEMBLE: 6.0,
            self.SIGNAL_FUSION_NN: 7.0,
            self.MULTI_TIMEFRAME_TREE: 5.5,
            
            # Custom (variable complexity)
            self.CUSTOM_TREE: 3.0,
            self.CUSTOM_NN: 5.0,
            self.HYBRID_ARCHITECTURE: 7.5
        }
        return complexity_map.get(self, 5.0)


@dataclass
class ClusteringAlgorithmConfig:
    """Configuration for clustering algorithms used in NAS."""
    
    # Algorithm selection
    algorithm: str = "kmeans"  # kmeans, hierarchical, dbscan, spectral, gaussian_mixture
    n_clusters: int = 5
    random_state: int = 42
    
    # K-means specific
    max_iter: int = 300
    tolerance: float = 1e-4
    n_init: int = 10
    
    # Hierarchical specific
    linkage: str = "ward"  # ward, complete, average, single
    distance_threshold: Optional[float] = None
    
    # DBSCAN specific
    eps: float = 0.5
    min_samples: int = 5
    
    # Spectral specific
    gamma: float = 1.0
    affinity: str = "rbf"  # rbf, polynomial, sigmoid, laplacian
    
    # Gaussian Mixture specific
    covariance_type: str = "full"  # full, tied, diag, spherical
    max_iter_em: int = 100
    
    # Validation
    validate_parameters: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.validate_parameters:
            self._validate()
    
    def _validate(self):
        """Validate clustering configuration parameters."""
        # Validate n_clusters
        if self.n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {self.n_clusters}")
        
        # Validate algorithm-specific parameters
        if self.algorithm == "kmeans":
            if self.max_iter <= 0:
                raise ValueError(f"max_iter must be positive, got {self.max_iter}")
            if self.tolerance <= 0:
                raise ValueError(f"tolerance must be positive, got {self.tolerance}")
            if self.n_init <= 0:
                raise ValueError(f"n_init must be positive, got {self.n_init}")
        
        elif self.algorithm == "hierarchical":
            valid_linkage = ["ward", "complete", "average", "single"]
            if self.linkage not in valid_linkage:
                raise ValueError(f"linkage must be one of {valid_linkage}, got {self.linkage}")
        
        elif self.algorithm == "dbscan":
            if self.eps <= 0:
                raise ValueError(f"eps must be positive, got {self.eps}")
            if self.min_samples <= 0:
                raise ValueError(f"min_samples must be positive, got {self.min_samples}")
        
        elif self.algorithm == "spectral":
            if self.gamma <= 0:
                raise ValueError(f"gamma must be positive, got {self.gamma}")
            valid_affinity = ["rbf", "polynomial", "sigmoid", "laplacian"]
            if self.affinity not in valid_affinity:
                raise ValueError(f"affinity must be one of {valid_affinity}, got {self.affinity}")
        
        elif self.algorithm == "gaussian_mixture":
            valid_covariance = ["full", "tied", "diag", "spherical"]
            if self.covariance_type not in valid_covariance:
                raise ValueError(f"covariance_type must be one of {valid_covariance}, got {self.covariance_type}")
            if self.max_iter_em <= 0:
                raise ValueError(f"max_iter_em must be positive, got {self.max_iter_em}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'algorithm': self.algorithm,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'tolerance': self.tolerance,
            'n_init': self.n_init,
            'linkage': self.linkage,
            'distance_threshold': self.distance_threshold,
            'eps': self.eps,
            'min_samples': self.min_samples,
            'gamma': self.gamma,
            'affinity': self.affinity,
            'covariance_type': self.covariance_type,
            'max_iter_em': self.max_iter_em,
            'validate_parameters': self.validate_parameters
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClusteringAlgorithmConfig':
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class ArchitectureSearchConfig:
    """Configuration for architecture search parameters."""
    
    # Search strategy
    search_strategy: str = "evolutionary"  # evolutionary, random, grid, bayesian, bayesian_tpe, random_forest
    max_generations: int = 50
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 2
    
    # Architecture constraints
    min_layers: int = 1
    max_layers: int = 10
    min_neurons: int = 10
    max_neurons: int = 1000
    activation_functions: List[str] = field(default_factory=lambda: ["relu", "tanh", "sigmoid"])
    
    # Tree-specific constraints
    min_depth: int = 1
    max_depth: int = 15
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[Union[str, int, float]] = None
    
    # Performance constraints
    max_training_time: float = 3600.0  # seconds
    max_memory_usage: float = 0.8  # fraction of available memory
    min_accuracy: float = 0.6
    max_overfitting_threshold: float = 0.1
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate()
    
    def _validate(self):
        """Validate architecture search parameters."""
        # Validate search parameters
        if self.max_generations <= 0:
            raise ValueError(f"max_generations must be positive, got {self.max_generations}")
        if self.population_size <= 0:
            raise ValueError(f"population_size must be positive, got {self.population_size}")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError(f"mutation_rate must be between 0 and 1, got {self.mutation_rate}")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError(f"crossover_rate must be between 0 and 1, got {self.crossover_rate}")
        if self.elite_size < 0 or self.elite_size >= self.population_size:
            raise ValueError(f"elite_size must be >= 0 and < population_size, got {self.elite_size}")
        
        # Validate architecture constraints
        if self.min_layers <= 0:
            raise ValueError(f"min_layers must be positive, got {self.min_layers}")
        if self.max_layers < self.min_layers:
            raise ValueError(f"max_layers must be >= min_layers, got {self.max_layers} < {self.min_layers}")
        if self.min_neurons <= 0:
            raise ValueError(f"min_neurons must be positive, got {self.min_neurons}")
        if self.max_neurons < self.min_neurons:
            raise ValueError(f"max_neurons must be >= min_neurons, got {self.max_neurons} < {self.min_neurons}")
        
        # Validate tree constraints
        if self.min_depth <= 0:
            raise ValueError(f"min_depth must be positive, got {self.min_depth}")
        if self.max_depth < self.min_depth:
            raise ValueError(f"max_depth must be >= min_depth, got {self.max_depth} < {self.min_depth}")
        if self.min_samples_split <= 0:
            raise ValueError(f"min_samples_split must be positive, got {self.min_samples_split}")
        if self.min_samples_leaf <= 0:
            raise ValueError(f"min_samples_leaf must be positive, got {self.min_samples_leaf}")
        
        # Validate performance constraints
        if self.max_training_time <= 0:
            raise ValueError(f"max_training_time must be positive, got {self.max_training_time}")
        if not 0 < self.max_memory_usage <= 1:
            raise ValueError(f"max_memory_usage must be between 0 and 1, got {self.max_memory_usage}")
        if not 0 <= self.min_accuracy <= 1:
            raise ValueError(f"min_accuracy must be between 0 and 1, got {self.min_accuracy}")
        if not 0 <= self.max_overfitting_threshold <= 1:
            raise ValueError(f"max_overfitting_threshold must be between 0 and 1, got {self.max_overfitting_threshold}")


@dataclass
class HardwareOptimizationConfig:
    """Configuration for hardware-specific optimizations."""
    
    # M1 optimization settings
    enable_m1_optimization: bool = True
    use_mps_acceleration: bool = True
    memory_limit_gb: Optional[float] = None
    cpu_optimization_level: str = "balanced"  # conservative, balanced, aggressive
    
    # Memory management
    enable_memory_monitoring: bool = True
    memory_checkpoint_frequency: int = 100  # iterations
    garbage_collection_frequency: int = 50  # iterations
    
    # Parallel processing
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None
    use_thread_pool: bool = True
    chunk_size: int = 1000
    
    # GPU settings (for non-M1 systems)
    enable_gpu_acceleration: bool = False
    gpu_memory_fraction: float = 0.8
    
    def __post_init__(self):
        """Validate hardware configuration."""
        if self.memory_limit_gb is not None and self.memory_limit_gb <= 0:
            raise ValueError(f"memory_limit_gb must be positive, got {self.memory_limit_gb}")
        if self.memory_checkpoint_frequency <= 0:
            raise ValueError(f"memory_checkpoint_frequency must be positive, got {self.memory_checkpoint_frequency}")
        if self.garbage_collection_frequency <= 0:
            raise ValueError(f"garbage_collection_frequency must be positive, got {self.garbage_collection_frequency}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if not 0 < self.gpu_memory_fraction <= 1:
            raise ValueError(f"gpu_memory_fraction must be between 0 and 1, got {self.gpu_memory_fraction}")
        
        valid_optimization_levels = ["conservative", "balanced", "aggressive"]
        if self.cpu_optimization_level not in valid_optimization_levels:
            raise ValueError(f"cpu_optimization_level must be one of {valid_optimization_levels}, got {self.cpu_optimization_level}")


class NASClusteringConfig:
    """
    Comprehensive Neural Architecture Search Clustering Configuration.
    
    This class provides a complete configuration system for NAS clustering operations,
    integrating with the existing utility modules for enhanced functionality.
    """
    
    def __init__(
        self,
        architecture_type: NASArchitectureType = NASArchitectureType.RANDOM_FOREST,
        clustering_config: Optional[ClusteringAlgorithmConfig] = None,
        search_config: Optional[ArchitectureSearchConfig] = None,
        hardware_config: Optional[HardwareOptimizationConfig] = None,
        data_config: Optional[Dict[str, Any]] = None,
        validation_config: Optional[Dict[str, Any]] = None,
        logging_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize NAS clustering configuration.
        
        Args:
            architecture_type: Type of neural architecture to search for
            clustering_config: Configuration for clustering algorithms
            search_config: Configuration for architecture search
            hardware_config: Configuration for hardware optimizations
            data_config: Configuration for data processing
            validation_config: Configuration for validation
            logging_config: Configuration for logging
        """
        self.architecture_type = architecture_type
        self.clustering_config = clustering_config or ClusteringAlgorithmConfig()
        self.search_config = search_config or ArchitectureSearchConfig()
        self.hardware_config = hardware_config or HardwareOptimizationConfig()
        
        # Data configuration with defaults
        self.data_config = {
            'feature_engineering': True,
            'normalization': 'standard',  # standard, minmax, robust, none
            'missing_value_strategy': 'impute',  # impute, drop, ignore
            'outlier_detection': True,
            'outlier_threshold': 3.0,
            'feature_selection': True,
            'max_features': 100,
            'correlation_threshold': 0.95,
            'variance_threshold': 0.01,
            **(data_config or {})
        }
        
        # Validation configuration with defaults
        self.validation_config = {
            'cross_validation_folds': 5,
            'test_size': 0.2,
            'validation_size': 0.2,
            'stratified_splits': True,
            'time_series_split': False,
            'purged_cv': False,
            'embargo_period': 0,
            'performance_metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
            'regression_metrics': ['mse', 'mae', 'r2', 'mape'],
            'early_stopping': True,
            'patience': 10,
            'min_delta': 0.001,
            **(validation_config or {})
        }
        
        # Logging configuration with defaults
        self.logging_config = {
            'log_level': 'INFO',
            'log_to_file': True,
            'log_file_path': 'nas_clustering.log',
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'enable_tprint': True,
            'log_architecture_evolution': True,
            'log_performance_metrics': True,
            'log_hardware_stats': True,
            **(logging_config or {})
        }
        
        # Initialize utility components
        self._initialize_utilities()
        
        # Setup logging
        self._setup_logging()
        
        # Validate configuration
        self._validate_configuration()
        
        tprint_success(f"NAS Clustering Configuration initialized for {architecture_type.name}")
    
    def _initialize_utilities(self):
        """Initialize utility components."""
        try:
            # Initialize hardware optimizers
            if HARDWARE_UTILS_AVAILABLE and self.hardware_config.enable_m1_optimization:
                self.gpu_manager = get_m1_gpu_manager()
                self.cpu_optimizer = get_m1_cpu_optimizer()
            else:
                self.gpu_manager = None
                self.cpu_optimizer = None
            
            if HARDWARE_UTILS_AVAILABLE and self.hardware_config.enable_memory_monitoring:
                self.memory_optimizer = get_m1_memory_optimizer(self.hardware_config.memory_limit_gb)
            else:
                self.memory_optimizer = None
            
            # Initialize serialization
            if SERIALIZATION_AVAILABLE:
                self.serializer = UniversalSerializer()
            else:
                self.serializer = UniversalSerializer()  # Use fallback
            
            # Initialize math validation
            if MATH_VALIDATION_AVAILABLE:
                self.math_validator = MathValidation()
            else:
                self.math_validator = MathValidation()  # Use fallback
            
            # Check hardware availability
            if HARDWARE_UTILS_AVAILABLE:
                self.is_m1_available = is_m1_available()
                self.is_mps_available = is_mps_available()
            else:
                self.is_m1_available = False
                self.is_mps_available = False
            
            if TPRINT_AVAILABLE:
                tprint_info(f"Hardware status - M1: {self.is_m1_available}, MPS: {self.is_mps_available}")
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_warning(f"Failed to initialize some utilities: {e}")
            # Set fallback values
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.serializer = UniversalSerializer()
            self.math_validator = MathValidation()
            self.is_m1_available = False
            self.is_mps_available = False
    
    def _setup_logging(self):
        """Setup logging configuration."""
        try:
            # Configure logging level
            log_level = getattr(logging, self.logging_config['log_level'].upper(), logging.INFO)
            logger.setLevel(log_level)
            
            # Setup file logging if enabled
            if self.logging_config['log_to_file']:
                log_file = Path(self.logging_config['log_file_path'])
                ensure_directory(log_file.parent)
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(log_level)
                
                formatter = logging.Formatter(self.logging_config['log_format'])
                file_handler.setFormatter(formatter)
                
                logger.addHandler(file_handler)
            
            tprint_info("Logging configuration applied successfully")
            
        except Exception as e:
            tprint_warning(f"Failed to setup logging: {e}")
    
    def _validate_configuration(self):
        """Validate the complete configuration."""
        try:
            # Validate architecture type
            if not isinstance(self.architecture_type, NASArchitectureType):
                raise ValueError(f"architecture_type must be NASArchitectureType, got {type(self.architecture_type)}")
            
            # Validate clustering configuration
            if not isinstance(self.clustering_config, ClusteringAlgorithmConfig):
                raise ValueError("clustering_config must be ClusteringAlgorithmConfig")
            
            # Validate search configuration
            if not isinstance(self.search_config, ArchitectureSearchConfig):
                raise ValueError("search_config must be ArchitectureSearchConfig")
            
            # Validate hardware configuration
            if not isinstance(self.hardware_config, HardwareOptimizationConfig):
                raise ValueError("hardware_config must be HardwareOptimizationConfig")
            
            # Validate data configuration
            required_data_keys = ['normalization', 'missing_value_strategy', 'outlier_threshold']
            for key in required_data_keys:
                if key not in self.data_config:
                    raise ValueError(f"Missing required data configuration key: {key}")
            
            # Validate validation configuration
            required_validation_keys = ['cross_validation_folds', 'test_size', 'performance_metrics']
            for key in required_validation_keys:
                if key not in self.validation_config:
                    raise ValueError(f"Missing required validation configuration key: {key}")
            
            tprint_success("Configuration validation passed")
            
        except Exception as e:
            tprint_error(f"Configuration validation failed: {e}")
            raise
    
    def optimize_for_hardware(self) -> Dict[str, Any]:
        """Optimize configuration for current hardware."""
        optimization_results = {
            'm1_optimization': False,
            'mps_acceleration': False,
            'memory_optimization': False,
            'cpu_optimization': False,
            'recommendations': []
        }
        
        try:
            # M1 optimization
            if self.is_m1_available and self.hardware_config.enable_m1_optimization:
                if self.cpu_optimizer:
                    cpu_result = self.cpu_optimizer.optimize_cpu_usage(
                        target_utilization=0.8,
                        aggressive=(self.hardware_config.cpu_optimization_level == "aggressive")
                    )
                    optimization_results['cpu_optimization'] = cpu_result.get('success', False)
                
                if self.memory_optimizer:
                    memory_result = self.memory_optimizer.optimize_memory_usage(
                        aggressive=(self.hardware_config.cpu_optimization_level == "aggressive")
                    )
                    optimization_results['memory_optimization'] = memory_result.get('success', False)
                
                optimization_results['m1_optimization'] = True
                optimization_results['recommendations'].append("M1 optimization enabled")
            
            # MPS acceleration
            if self.is_mps_available and self.hardware_config.use_mps_acceleration:
                optimization_results['mps_acceleration'] = True
                optimization_results['recommendations'].append("MPS acceleration available")
            
            # Memory monitoring
            if self.hardware_config.enable_memory_monitoring and self.memory_optimizer:
                self.memory_optimizer.start_monitoring()
                optimization_results['recommendations'].append("Memory monitoring started")
            
            tprint_success("Hardware optimization completed")
            
        except Exception as e:
            tprint_warning(f"Hardware optimization failed: {e}")
            optimization_results['error'] = str(e)
        
        return optimization_results
    
    def validate_data(self, data: Union[Any, None]) -> Dict[str, Any]:
        """Validate input data for NAS clustering."""
        validation_results = {
            'is_valid': True,
            'data_shape': None,
            'data_type': None,
            'quality_metrics': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Handle None or empty data
            if data is None:
                validation_results['is_valid'] = False
                validation_results['issues'].append("Data is None")
                return validation_results
            
            # Convert to DataFrame if pandas is available
            if PANDAS_AVAILABLE and pd is not None:
                if isinstance(data, np.ndarray) if NUMPY_AVAILABLE else isinstance(data, (list, tuple)):
                    df = pd.DataFrame(data)
                    validation_results['data_type'] = 'numpy_array'
                elif hasattr(data, 'shape') and hasattr(data, 'columns'):
                    df = data
                    validation_results['data_type'] = 'dataframe'
                else:
                    # Try to convert to DataFrame
                    df = pd.DataFrame(data)
                    validation_results['data_type'] = 'converted_dataframe'
                
                validation_results['data_shape'] = df.shape
                
                # Basic validation
                if df.empty:
                    validation_results['is_valid'] = False
                    validation_results['issues'].append("Data is empty")
                    return validation_results
                
                # Check for required columns
                required_columns = ['features'] if hasattr(df, 'columns') and 'features' in df.columns else []
                if required_columns:
                    missing_columns = set(required_columns) - set(df.columns)
                    if missing_columns:
                        validation_results['issues'].append(f"Missing required columns: {missing_columns}")
                
                # Calculate quality metrics
                if COMMON_UTILITIES_AVAILABLE:
                    quality_metrics = calculate_data_quality_metrics(df)
                else:
                    quality_metrics = {'total_rows': len(df)}
                validation_results['quality_metrics'] = quality_metrics
                
                # Check for issues
                if 'missing_percentage' in quality_metrics and quality_metrics['missing_percentage'] > 50:
                    validation_results['issues'].append("High percentage of missing values")
                
                if 'duplicate_percentage' in quality_metrics and quality_metrics['duplicate_percentage'] > 10:
                    validation_results['issues'].append("High percentage of duplicate rows")
                
                # Architecture-specific validation
                if self.architecture_type.is_tree_based():
                    if 'numeric_columns' in quality_metrics and quality_metrics['numeric_columns'] == 0:
                        validation_results['issues'].append("No numeric columns found for tree-based architecture")
                
                # Generate recommendations
                if 'missing_percentage' in quality_metrics and quality_metrics['missing_percentage'] > 20:
                    validation_results['recommendations'].append("Consider imputing missing values")
                
                if 'numeric_columns' in quality_metrics and quality_metrics['numeric_columns'] > 100:
                    validation_results['recommendations'].append("Consider feature selection")
            
            else:
                # Fallback for when pandas is not available
                validation_results['data_type'] = 'raw_data'
                validation_results['data_shape'] = (len(data),) if hasattr(data, '__len__') else (1,)
                validation_results['quality_metrics'] = {'total_rows': len(data) if hasattr(data, '__len__') else 1}
            
            # Overall validation
            if validation_results['issues']:
                validation_results['is_valid'] = False
            
            if TPRINT_AVAILABLE:
                tprint_info(f"Data validation completed - Valid: {validation_results['is_valid']}")
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"Data validation failed: {e}")
            validation_results['is_valid'] = False
            validation_results['error'] = str(e)
        
        return validation_results
    
    def get_architecture_complexity(self) -> Dict[str, Any]:
        """Get complexity analysis for the selected architecture type."""
        complexity_info = {
            'architecture_type': self.architecture_type.name,
            'complexity_factor': self.architecture_type.get_complexity_factor(),
            'is_tree_based': self.architecture_type.is_tree_based(),
            'is_neural_network': self.architecture_type.is_neural_network(),
            'is_ensemble': self.architecture_type.is_ensemble(),
            'is_trading_specific': self.architecture_type.is_trading_specific(),
            'estimated_training_time': 0.0,
            'estimated_memory_usage': 0.0,
            'recommended_workers': 1
        }
        
        try:
            # Estimate training time based on complexity
            base_time = 60.0  # seconds
            complexity_multiplier = self.architecture_type.get_complexity_factor()
            complexity_info['estimated_training_time'] = base_time * complexity_multiplier
            
            # Estimate memory usage
            if self.architecture_type.is_tree_based():
                base_memory = 100  # MB
            elif self.architecture_type.is_neural_network():
                base_memory = 500  # MB
            else:
                base_memory = 300  # MB
            
            complexity_info['estimated_memory_usage'] = base_memory * complexity_multiplier
            
            # Recommend number of workers
            if self.hardware_config.enable_parallel_processing:
                if self.is_m1_available and self.cpu_optimizer:
                    complexity_info['recommended_workers'] = self.cpu_optimizer.get_optimal_worker_count()
                else:
                    complexity_info['recommended_workers'] = min(4, self.search_config.population_size // 5)
            
            tprint_info(f"Architecture complexity: {complexity_info['complexity_factor']:.2f}")
            
        except Exception as e:
            tprint_warning(f"Complexity analysis failed: {e}")
            complexity_info['error'] = str(e)
        
        return complexity_info
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'architecture_type': {
                'name': self.architecture_type.name,
                'value': self.architecture_type.value
            },
            'clustering_config': self.clustering_config.to_dict(),
            'search_config': {
                'search_strategy': self.search_config.search_strategy,
                'max_generations': self.search_config.max_generations,
                'population_size': self.search_config.population_size,
                'mutation_rate': self.search_config.mutation_rate,
                'crossover_rate': self.search_config.crossover_rate,
                'elite_size': self.search_config.elite_size,
                'min_layers': self.search_config.min_layers,
                'max_layers': self.search_config.max_layers,
                'min_neurons': self.search_config.min_neurons,
                'max_neurons': self.search_config.max_neurons,
                'activation_functions': self.search_config.activation_functions,
                'min_depth': self.search_config.min_depth,
                'max_depth': self.search_config.max_depth,
                'min_samples_split': self.search_config.min_samples_split,
                'min_samples_leaf': self.search_config.min_samples_leaf,
                'max_features': self.search_config.max_features,
                'max_training_time': self.search_config.max_training_time,
                'max_memory_usage': self.search_config.max_memory_usage,
                'min_accuracy': self.search_config.min_accuracy,
                'max_overfitting_threshold': self.search_config.max_overfitting_threshold,
                'early_stopping_patience': self.search_config.early_stopping_patience,
                'early_stopping_min_delta': self.search_config.early_stopping_min_delta
            },
            'hardware_config': {
                'enable_m1_optimization': self.hardware_config.enable_m1_optimization,
                'use_mps_acceleration': self.hardware_config.use_mps_acceleration,
                'memory_limit_gb': self.hardware_config.memory_limit_gb,
                'cpu_optimization_level': self.hardware_config.cpu_optimization_level,
                'enable_memory_monitoring': self.hardware_config.enable_memory_monitoring,
                'memory_checkpoint_frequency': self.hardware_config.memory_checkpoint_frequency,
                'garbage_collection_frequency': self.hardware_config.garbage_collection_frequency,
                'enable_parallel_processing': self.hardware_config.enable_parallel_processing,
                'max_workers': self.hardware_config.max_workers,
                'use_thread_pool': self.hardware_config.use_thread_pool,
                'chunk_size': self.hardware_config.chunk_size,
                'enable_gpu_acceleration': self.hardware_config.enable_gpu_acceleration,
                'gpu_memory_fraction': self.hardware_config.gpu_memory_fraction
            },
            'data_config': self.data_config,
            'validation_config': self.validation_config,
            'logging_config': self.logging_config,
            'timestamp': format_datetime(get_current_datetime()),
            'version': '1.0.0'
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NASClusteringConfig':
        """Create configuration from dictionary."""
        try:
            # Extract architecture type
            arch_type_info = config_dict.get('architecture_type', {})
            if isinstance(arch_type_info, dict):
                arch_type_name = arch_type_info.get('name', 'RANDOM_FOREST')
                architecture_type = getattr(NASArchitectureType, arch_type_name, NASArchitectureType.RANDOM_FOREST)
            else:
                architecture_type = NASArchitectureType.RANDOM_FOREST
            
            # Extract configurations
            clustering_config = ClusteringAlgorithmConfig.from_dict(
                config_dict.get('clustering_config', {})
            )
            
            search_config_dict = config_dict.get('search_config', {})
            search_config = ArchitectureSearchConfig(**search_config_dict)
            
            hardware_config_dict = config_dict.get('hardware_config', {})
            hardware_config = HardwareOptimizationConfig(**hardware_config_dict)
            
            # Extract other configurations
            data_config = config_dict.get('data_config', {})
            validation_config = config_dict.get('validation_config', {})
            logging_config = config_dict.get('logging_config', {})
            
            return cls(
                architecture_type=architecture_type,
                clustering_config=clustering_config,
                search_config=search_config,
                hardware_config=hardware_config,
                data_config=data_config,
                validation_config=validation_config,
                logging_config=logging_config
            )
            
        except Exception as e:
            tprint_error(f"Failed to create configuration from dictionary: {e}")
            raise
    
    def save_to_file(self, filepath: Union[str, Path]) -> bool:
        """Save configuration to file."""
        try:
            filepath = Path(filepath)
            ensure_directory(filepath.parent)
            
            config_dict = self.to_dict()
            
            if filepath.suffix.lower() == '.json':
                success = safe_json_dump(config_dict, filepath, indent=2)
            elif filepath.suffix.lower() in ['.pkl', '.pickle']:
                success = self.serializer.save(config_dict, str(filepath), 'pickle')
            else:
                # Default to JSON
                success = safe_json_dump(config_dict, filepath, indent=2)
            
            if success:
                tprint_success(f"Configuration saved to: {filepath}")
            else:
                tprint_error(f"Failed to save configuration to: {filepath}")
            
            return success
            
        except Exception as e:
            tprint_error(f"Error saving configuration: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'NASClusteringConfig':
        """Load configuration from file."""
        try:
            filepath = Path(filepath)
            
            if not safe_file_exists(filepath):
                raise FileNotFoundError(f"Configuration file not found: {filepath}")
            
            if filepath.suffix.lower() == '.json':
                config_dict = safe_json_load(filepath)
            elif filepath.suffix.lower() in ['.pkl', '.pickle']:
                config_dict = PickleSerializer.load(str(filepath))
            else:
                # Try JSON first, then pickle
                config_dict = safe_json_load(filepath)
                if config_dict is None:
                    config_dict = PickleSerializer.load(str(filepath))
            
            if config_dict is None:
                raise ValueError(f"Failed to load configuration from: {filepath}")
            
            tprint_success(f"Configuration loaded from: {filepath}")
            return cls.from_dict(config_dict)
            
        except Exception as e:
            tprint_error(f"Error loading configuration: {e}")
            raise
    
    def cleanup_resources(self):
        """Cleanup resources and stop monitoring."""
        try:
            if self.memory_optimizer:
                self.memory_optimizer.stop_monitoring()
                tprint_info("Memory monitoring stopped")
            
            # Close any file handlers
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
            
            tprint_success("Resources cleaned up successfully")
            
        except Exception as e:
            tprint_warning(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup_resources()
        except (AttributeError, TypeError) as e:
            tprint(f"Error in cleanup: {e}", level="error")


# Configuration presets
def create_default_config() -> NASClusteringConfig:
    """Create default NAS clustering configuration."""
    return NASClusteringConfig()


def create_development_config() -> NASClusteringConfig:
    """Create development configuration with relaxed constraints."""
    clustering_config = ClusteringAlgorithmConfig(
        algorithm="kmeans",
        n_clusters=3,
        max_iter=100,
        validate_parameters=True
    )
    
    search_config = ArchitectureSearchConfig(
        search_strategy="random",
        max_generations=10,
        population_size=10,
        max_layers=5,
        max_neurons=100,
        max_training_time=300.0,
        early_stopping_patience=5
    )
    
    hardware_config = HardwareOptimizationConfig(
        enable_m1_optimization=True,
        enable_memory_monitoring=False,
        cpu_optimization_level="conservative",
        max_workers=2
    )
    
    return NASClusteringConfig(
        architecture_type=NASArchitectureType.RANDOM_FOREST,
        clustering_config=clustering_config,
        search_config=search_config,
        hardware_config=hardware_config,
        logging_config={'log_level': 'DEBUG', 'enable_tprint': True}
    )


def create_production_config() -> NASClusteringConfig:
    """Create production configuration with strict constraints."""
    clustering_config = ClusteringAlgorithmConfig(
        algorithm="hierarchical",
        n_clusters=8,
        max_iter=500,
        validate_parameters=True
    )
    
    search_config = ArchitectureSearchConfig(
        search_strategy="evolutionary",
        max_generations=100,
        population_size=50,
        max_layers=15,
        max_neurons=2000,
        max_training_time=7200.0,
        early_stopping_patience=20
    )
    
    hardware_config = HardwareOptimizationConfig(
        enable_m1_optimization=True,
        enable_memory_monitoring=True,
        memory_limit_gb=16.0,
        cpu_optimization_level="aggressive",
        max_workers=None  # Auto-detect
    )
    
    return NASClusteringConfig(
        architecture_type=NASArchitectureType.TRADING_ENSEMBLE,
        clustering_config=clustering_config,
        search_config=search_config,
        hardware_config=hardware_config,
        logging_config={'log_level': 'INFO', 'enable_tprint': True}
    )


def create_trading_config() -> NASClusteringConfig:
    """Create trading-specific configuration."""
    clustering_config = ClusteringAlgorithmConfig(
        algorithm="gaussian_mixture",
        n_clusters=5,
        covariance_type="full",
        validate_parameters=True
    )
    
    search_config = ArchitectureSearchConfig(
        search_strategy="bayesian",
        max_generations=75,
        population_size=30,
        max_layers=12,
        max_neurons=1500,
        max_training_time=5400.0,
        early_stopping_patience=15
    )
    
    hardware_config = HardwareOptimizationConfig(
        enable_m1_optimization=True,
        enable_memory_monitoring=True,
        memory_limit_gb=12.0,
        cpu_optimization_level="balanced",
        enable_parallel_processing=True
    )
    
    data_config = {
        'feature_engineering': True,
        'normalization': 'robust',
        'outlier_detection': True,
        'outlier_threshold': 2.5,
        'feature_selection': True,
        'max_features': 50,
        'correlation_threshold': 0.9
    }
    
    validation_config = {
        'cross_validation_folds': 3,
        'time_series_split': True,
        'purged_cv': True,
        'embargo_period': 1,
        'performance_metrics': ['sharpe_ratio', 'max_drawdown', 'profit_factor', 'win_rate']
    }
    
    return NASClusteringConfig(
        architecture_type=NASArchitectureType.REGIME_AWARE_TREE,
        clustering_config=clustering_config,
        search_config=search_config,
        hardware_config=hardware_config,
        data_config=data_config,
        validation_config=validation_config,
        logging_config={'log_level': 'INFO', 'enable_tprint': True}
    )


# Export main classes and functions
__all__ = [
    'NASArchitectureType',
    'ClusteringAlgorithmConfig', 
    'ArchitectureSearchConfig',
    'HardwareOptimizationConfig',
    'NASClusteringConfig',
    'create_default_config',
    'create_development_config', 
    'create_production_config',
    'create_trading_config'
]