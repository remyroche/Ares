"""
NAS Configuration - Neural Architecture Search Configuration System

This module provides comprehensive configuration classes for NAS clustering,
including architecture types, clustering parameters, optimization settings,
and hardware-specific configurations.

Key Features:
- Flexible architecture type definitions
- Comprehensive clustering configuration
- Hardware-aware optimization settings
- Validation and constraint checking
- Serialization and deserialization support
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path

# Import shared utilities
from src.utils.common_operations import safe_dataframe_operation, validate_dataframe_columns
from src.utils.math_validation import validate_finite, validate_positive, validate_range
from src.utils.serialization_utils import save_object, load_object
from src.utils.tprint import tprint

# Setup logging
logger = logging.getLogger(__name__)

class NASArchitectureType(Enum):
    """Enumeration of supported neural architecture types."""
    FEEDFORWARD = "feedforward"
    CONVOLUTIONAL = "convolutional"
    RECURRENT = "recurrent"
    TRANSFORMER = "transformer"
    RESIDUAL = "residual"
    ATTENTION = "attention"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"

class ClusteringAlgorithm(Enum):
    """Enumeration of supported clustering algorithms."""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    SPECTRAL = "spectral"
    BIRCH = "birch"
    MEAN_SHIFT = "mean_shift"
    OPTICS = "optics"

class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies."""
    BAYESIAN = "bayesian"
    GRID = "grid"
    RANDOM = "random"
    EVOLUTIONARY = "evolutionary"
    GRADIENT = "gradient"
    HYBRID = "hybrid"

class HardwareAcceleration(Enum):
    """Enumeration of hardware acceleration options."""
    CPU = "cpu"
    GPU = "gpu"
    M1_GPU = "m1_gpu"
    M1_CPU = "m1_cpu"
    AUTO = "auto"
    NONE = "none"

@dataclass
class ArchitectureConstraints:
    """Constraints for neural architecture design."""
    min_layers: int = 1
    max_layers: int = 10
    min_neurons: int = 10
    max_neurons: int = 1000
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1.0
    min_dropout: float = 0.0
    max_dropout: float = 0.8
    allowed_activations: List[str] = field(default_factory=lambda: ['relu', 'tanh', 'sigmoid', 'leaky_relu'])
    allowed_optimizers: List[str] = field(default_factory=lambda: ['adam', 'sgd', 'rmsprop', 'adamw'])
    allowed_loss_functions: List[str] = field(default_factory=lambda: ['mse', 'mae', 'huber', 'log_cosh'])

@dataclass
class ClusteringConstraints:
    """Constraints for clustering parameters."""
    min_clusters: int = 2
    max_clusters: int = 20
    min_samples: int = 1
    max_samples: int = 10000
    min_eps: float = 0.1
    max_eps: float = 2.0
    allowed_linkage: List[str] = field(default_factory=lambda: ['ward', 'complete', 'average', 'single'])
    allowed_metrics: List[str] = field(default_factory=lambda: ['euclidean', 'manhattan', 'cosine', 'chebyshev'])

@dataclass
class OptimizationConstraints:
    """Constraints for optimization parameters."""
    min_trials: int = 10
    max_trials: int = 1000
    min_timeout: int = 60  # seconds
    max_timeout: int = 86400  # 24 hours
    min_early_stopping: int = 5
    max_early_stopping: int = 100
    min_parallel_trials: int = 1
    max_parallel_trials: int = 32

@dataclass
class HardwareConstraints:
    """Constraints for hardware-specific parameters."""
    min_memory_gb: float = 1.0
    max_memory_gb: float = 64.0
    min_batch_size: int = 1
    max_batch_size: int = 1024
    min_workers: int = 1
    max_workers: int = 16
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = False

@dataclass
class NASClusteringConfig:
    """
    Comprehensive configuration for NAS clustering system.
    
    This class provides all necessary configuration parameters for:
    - Neural architecture search
    - Clustering algorithms
    - Optimization strategies
    - Hardware acceleration
    - Data processing
    - Validation and monitoring
    """
    
    # Basic configuration
    name: str = "NASClusteringConfig"
    version: str = "1.0.0"
    description: str = "Neural Architecture Search Clustering Configuration"
    
    # Architecture configuration
    architecture_type: NASArchitectureType = NASArchitectureType.FEEDFORWARD
    architecture_constraints: ArchitectureConstraints = field(default_factory=ArchitectureConstraints)
    
    # Clustering configuration
    clustering_algorithm: ClusteringAlgorithm = ClusteringAlgorithm.KMEANS
    n_clusters: int = 5
    clustering_constraints: ClusteringConstraints = field(default_factory=ClusteringConstraints)
    
    # Optimization configuration
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN
    n_trials: int = 100
    timeout: int = 3600  # seconds
    early_stopping_rounds: int = 10
    parallel_trials: int = 4
    optimization_constraints: OptimizationConstraints = field(default_factory=OptimizationConstraints)
    
    # Hardware configuration
    hardware_acceleration: HardwareAcceleration = HardwareAcceleration.AUTO
    memory_limit_gb: Optional[float] = None
    batch_size: int = 32
    num_workers: int = 4
    hardware_constraints: HardwareConstraints = field(default_factory=HardwareConstraints)
    
    # Data processing configuration
    preprocessing_enabled: bool = True
    feature_scaling: str = "standard"  # standard, minmax, robust, none
    outlier_detection: bool = True
    missing_value_strategy: str = "interpolate"  # interpolate, drop, fill
    data_validation: bool = True
    
    # Cross-validation configuration
    cv_folds: int = 5
    cv_strategy: str = "kfold"  # kfold, stratified, time_series
    test_size: float = 0.2
    validation_size: float = 0.2
    
    # Lookahead bias prevention
    lookahead_steps: int = 10
    lookahead_detection: bool = True
    lookahead_penalty: float = 0.1
    
    # Hyperparameter optimization
    hyperparameter_optimization: bool = True
    hpo_algorithm: str = "tpe"  # tpe, random, grid, bayesian
    hpo_trials: int = 50
    hpo_timeout: int = 1800  # 30 minutes
    
    # Ensemble configuration
    ensemble_enabled: bool = True
    ensemble_methods: List[str] = field(default_factory=lambda: ['voting', 'averaging', 'stacking'])
    ensemble_weights: Optional[List[float]] = None
    
    # Multi-objective optimization
    multi_objective: bool = True
    objectives: List[str] = field(default_factory=lambda: ['accuracy', 'complexity', 'speed'])
    objective_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    
    # Adaptive configuration
    adaptive_clustering: bool = True
    adaptive_learning_rate: bool = True
    adaptive_batch_size: bool = True
    
    # Online learning
    online_learning: bool = False
    incremental_update: bool = True
    forgetting_factor: float = 0.9
    
    # Monitoring and logging
    verbose: bool = True
    log_level: str = "INFO"
    save_intermediate_results: bool = True
    save_final_model: bool = True
    output_directory: str = "nas_clustering_results"
    
    # Performance monitoring
    performance_monitoring: bool = True
    memory_monitoring: bool = True
    gpu_monitoring: bool = True
    execution_time_monitoring: bool = True
    
    # Quality assurance
    quality_checks: bool = True
    data_quality_threshold: float = 0.8
    model_quality_threshold: float = 0.7
    convergence_threshold: float = 1e-4
    
    # Advanced features
    transfer_learning: bool = False
    pretrained_models: List[str] = field(default_factory=list)
    fine_tuning: bool = False
    regularization: bool = True
    dropout_rate: float = 0.1
    weight_decay: float = 1e-4
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_configuration()
        self._setup_directories()
        self._configure_logging()
    
    def _validate_configuration(self):
        """Validate configuration parameters."""
        try:
            # Validate basic parameters
            validate_positive(self.n_clusters, "n_clusters")
            validate_range(self.n_clusters, self.clustering_constraints.min_clusters, 
                          self.clustering_constraints.max_clusters, "n_clusters")
            
            validate_positive(self.n_trials, "n_trials")
            validate_range(self.n_trials, self.optimization_constraints.min_trials,
                          self.optimization_constraints.max_trials, "n_trials")
            
            validate_positive(self.timeout, "timeout")
            validate_range(self.timeout, self.optimization_constraints.min_timeout,
                          self.optimization_constraints.max_timeout, "timeout")
            
            # Validate hardware parameters
            if self.memory_limit_gb is not None:
                validate_positive(self.memory_limit_gb, "memory_limit_gb")
                validate_range(self.memory_limit_gb, self.hardware_constraints.min_memory_gb,
                              self.hardware_constraints.max_memory_gb, "memory_limit_gb")
            
            validate_positive(self.batch_size, "batch_size")
            validate_range(self.batch_size, self.hardware_constraints.min_batch_size,
                          self.hardware_constraints.max_batch_size, "batch_size")
            
            validate_positive(self.num_workers, "num_workers")
            validate_range(self.num_workers, self.hardware_constraints.min_workers,
                          self.hardware_constraints.max_workers, "num_workers")
            
            # Validate cross-validation parameters
            validate_range(self.test_size, 0.0, 1.0, "test_size")
            validate_range(self.validation_size, 0.0, 1.0, "validation_size")
            
            if self.test_size + self.validation_size >= 1.0:
                raise ValueError("test_size + validation_size must be less than 1.0")
            
            # Validate objective weights
            if len(self.objective_weights) != len(self.objectives):
                raise ValueError("objective_weights length must match objectives length")
            
            if abs(sum(self.objective_weights) - 1.0) > 1e-6:
                raise ValueError("objective_weights must sum to 1.0")
            
            tprint("✅ Configuration validation completed successfully")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}")
    
    def _setup_directories(self):
        """Setup output directories."""
        try:
            output_path = Path(self.output_directory)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (output_path / "models").mkdir(exist_ok=True)
            (output_path / "results").mkdir(exist_ok=True)
            (output_path / "logs").mkdir(exist_ok=True)
            (output_path / "plots").mkdir(exist_ok=True)
            
            tprint(f"📁 Output directories created: {output_path}")
            
        except Exception as e:
            logger.warning(f"Directory setup failed: {e}")
    
    def _configure_logging(self):
        """Configure logging based on configuration."""
        try:
            if self.verbose:
                # Configure logging level
                log_level = getattr(logging, self.log_level.upper(), logging.INFO)
                logger.setLevel(log_level)
                
                # Create file handler if output directory exists
                if os.path.exists(self.output_directory):
                    log_file = os.path.join(self.output_directory, "logs", f"{self.name}.log")
                    file_handler = logging.FileHandler(log_file)
                    file_handler.setLevel(log_level)
                    
                    # Create formatter
                    formatter = logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                
                tprint(f"📝 Logging configured: {self.log_level}")
            
        except Exception as e:
            logger.warning(f"Logging configuration failed: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        try:
            config_dict = {}
            for field_name, field_value in self.__dict__.items():
                if isinstance(field_value, Enum):
                    config_dict[field_name] = field_value.value
                elif hasattr(field_value, '__dict__'):
                    config_dict[field_name] = field_value.__dict__
                else:
                    config_dict[field_name] = field_value
            
            return config_dict
            
        except Exception as e:
            logger.error(f"Configuration serialization failed: {e}")
            return {}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NASClusteringConfig':
        """Create configuration from dictionary."""
        try:
            # Handle enum conversions
            if 'architecture_type' in config_dict:
                config_dict['architecture_type'] = NASArchitectureType(config_dict['architecture_type'])
            
            if 'clustering_algorithm' in config_dict:
                config_dict['clustering_algorithm'] = ClusteringAlgorithm(config_dict['clustering_algorithm'])
            
            if 'optimization_strategy' in config_dict:
                config_dict['optimization_strategy'] = OptimizationStrategy(config_dict['optimization_strategy'])
            
            if 'hardware_acceleration' in config_dict:
                config_dict['hardware_acceleration'] = HardwareAcceleration(config_dict['hardware_acceleration'])
            
            # Handle nested dataclasses
            if 'architecture_constraints' in config_dict:
                config_dict['architecture_constraints'] = ArchitectureConstraints(**config_dict['architecture_constraints'])
            
            if 'clustering_constraints' in config_dict:
                config_dict['clustering_constraints'] = ClusteringConstraints(**config_dict['clustering_constraints'])
            
            if 'optimization_constraints' in config_dict:
                config_dict['optimization_constraints'] = OptimizationConstraints(**config_dict['optimization_constraints'])
            
            if 'hardware_constraints' in config_dict:
                config_dict['hardware_constraints'] = HardwareConstraints(**config_dict['hardware_constraints'])
            
            return cls(**config_dict)
            
        except Exception as e:
            logger.error(f"Configuration deserialization failed: {e}")
            raise
    
    def save(self, filepath: str) -> bool:
        """Save configuration to file."""
        try:
            config_dict = self.to_dict()
            success = save_object(config_dict, filepath)
            
            if success:
                tprint(f"💾 Configuration saved to {filepath}")
            else:
                tprint(f"❌ Failed to save configuration to {filepath}")
            
            return success
            
        except Exception as e:
            logger.error(f"Configuration saving failed: {e}")
            return False
    
    @classmethod
    def load(cls, filepath: str) -> 'NASClusteringConfig':
        """Load configuration from file."""
        try:
            config_dict = load_object(filepath)
            
            if config_dict is None:
                raise ValueError(f"Could not load configuration from {filepath}")
            
            config = cls.from_dict(config_dict)
            tprint(f"📁 Configuration loaded from {filepath}")
            
            return config
            
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise
    
    def update(self, **kwargs) -> 'NASClusteringConfig':
        """Update configuration with new parameters."""
        try:
            # Create new configuration with updated parameters
            current_dict = self.to_dict()
            current_dict.update(kwargs)
            
            # Validate updated configuration
            updated_config = self.from_dict(current_dict)
            updated_config._validate_configuration()
            
            tprint("✅ Configuration updated successfully")
            return updated_config
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the configuration."""
        return {
            'name': self.name,
            'version': self.version,
            'architecture_type': self.architecture_type.value,
            'clustering_algorithm': self.clustering_algorithm.value,
            'optimization_strategy': self.optimization_strategy.value,
            'hardware_acceleration': self.hardware_acceleration.value,
            'n_clusters': self.n_clusters,
            'n_trials': self.n_trials,
            'timeout': self.timeout,
            'parallel_trials': self.parallel_trials,
            'verbose': self.verbose,
            'output_directory': self.output_directory
        }
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"NASClusteringConfig(name='{self.name}', architecture_type='{self.architecture_type.value}', clustering_algorithm='{self.clustering_algorithm.value}')"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()

# Convenience functions
def create_default_config() -> NASClusteringConfig:
    """Create a default configuration."""
    return NASClusteringConfig()

def create_high_performance_config() -> NASClusteringConfig:
    """Create a high-performance configuration."""
    return NASClusteringConfig(
        n_trials=500,
        timeout=7200,
        parallel_trials=8,
        hardware_acceleration=HardwareAcceleration.GPU,
        memory_limit_gb=16.0,
        batch_size=64,
        num_workers=8,
        verbose=True
    )

def create_fast_config() -> NASClusteringConfig:
    """Create a fast configuration for quick testing."""
    return NASClusteringConfig(
        n_trials=20,
        timeout=300,
        parallel_trials=2,
        hardware_acceleration=HardwareAcceleration.CPU,
        batch_size=16,
        num_workers=2,
        verbose=False
    )

# Export main classes and functions
__all__ = [
    'NASClusteringConfig',
    'NASArchitectureType',
    'ClusteringAlgorithm',
    'OptimizationStrategy',
    'HardwareAcceleration',
    'ArchitectureConstraints',
    'ClusteringConstraints',
    'OptimizationConstraints',
    'HardwareConstraints',
    'create_default_config',
    'create_high_performance_config',
    'create_fast_config'
]